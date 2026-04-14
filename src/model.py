# model.py
import torch
import torch_geometric
from torch_geometric.nn import LayerNorm 
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm
from attention import PairBiasedSelfAttention, SigmoidCrossAttention, GatedFusion, EnhancedBidirectionalCrossAttentionV2, drug_gaussian_position_embedding
from torch_scatter import scatter_std
from torch_geometric.utils import to_dense_batch


class DrugGVPModel(nn.Module):
    def __init__(self, 
        node_in_dim=[66, 1], node_h_dim=[128, 64],
        edge_in_dim=[16, 1], edge_h_dim=[32, 1],
        num_layers=3, drop_rate=0.1
    ):
        """
        Parameters
        ----------
        node_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        node_h_dims : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        """
        super(DrugGVPModel, self).__init__()
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, xd):
        # Unpack input data
        h_V = (xd.node_s, xd.node_v)
        h_E = (xd.edge_s, xd.edge_v)
        edge_index = xd.edge_index
        batch = xd.batch

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        molecule_embedding = out
        # per-graph mean
        out = torch_geometric.nn.global_add_pool(out, batch)

        return out, molecule_embedding

    


class Prot3DGraphModel(nn.Module):
    def __init__(self,
        d_vocab=21, d_embed=20,
        d_dihedrals=6, # <-- 6 + 18 new scalars
        d_pretrained_emb=1280, d_edge=39,
        d_gcn=[128, 256, 256],
    ):
        super(Prot3DGraphModel, self).__init__()
        d_gcn_in = d_gcn[0]
        self.embed = nn.Embedding(d_vocab, d_embed)
        self.proj_node = nn.Linear(d_embed + d_dihedrals + d_pretrained_emb, d_gcn_in)
        self.proj_edge = nn.Linear(d_edge, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        layers = []
        for i in range(len(gcn_layer_sizes) - 1):            
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
                'x, edge_index, edge_attr -> x'
            ))            
            layers.append(nn.LeakyReLU())            
        
        self.gcn = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr', layers)        
        self.pool = torch_geometric.nn.global_mean_pool
        

    def forward(self, data):
        x, edge_index = data.seq, data.edge_index
        batch = data.batch

        x = self.embed(x)
        s = data.node_s
        emb = data.seq_emb
        x = torch.cat([x, s, emb], dim=-1)

        edge_attr = data.edge_s

        x = self.proj_node(x)
        edge_attr = self.proj_edge(edge_attr)

        embeddings = self.gcn(x, edge_index, edge_attr)
        x = torch_geometric.nn.global_mean_pool(embeddings, batch)
        return x, embeddings


class MeanMaxStdPooling(nn.Module):
    """
    Concatenates mean, max, and optionally std pooling.
    Projects back to original dimension to maintain compatibility.
    """
    def __init__(self, input_dim, use_std=True, output_dim=None):
        super().__init__()
        self.use_std = use_std
        
        # Calculate concatenated dimension
        if use_std:
            self.concat_dim = input_dim * 3  # mean + max + std
        else:
            self.concat_dim = input_dim * 2  # mean + max only
        
        # Output dimension (defaults to input_dim for drop-in replacement)
        if output_dim is None:
            output_dim = input_dim
        
        # Projection layer to combine the concatenated features
        self.projection = nn.Sequential(
            nn.Linear(self.concat_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU()
        )

    def forward(self, x, batch):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch assignment [num_nodes]
        Returns:
            Pooled features [batch_size, output_dim]
        """
        # Mean pooling
        x_mean = global_mean_pool(x, batch)  # [batch_size, input_dim]
        
        # Max pooling
        # x_max = global_max_pool(x, batch)    # [batch_size, input_dim]
        x_dense, mask = to_dense_batch(x, batch)  # [B, Nmax, F], [B, Nmax]
        NEG = torch.finfo(x.dtype).min  # safe for fp32/fp16/bf16
        x_masked = x_dense.masked_fill(~mask.unsqueeze(-1), NEG)
        x_max = x_masked.amax(dim=1)  # [B, F]
        # Concatenate
        if self.use_std:
            # Std pooling
            x_std = scatter_std(x, batch, dim=0)  # [batch_size, input_dim]
            pooled = torch.cat([x_mean, x_max, x_std], dim=-1)
        else:
            pooled = torch.cat([x_mean, x_max], dim=-1)
        
        # Project to output dimension
        output = self.projection(pooled)
        
        return output
    
class Bidirection_Attention_Model(nn.Module):
    def __init__(self,
            prot_emb_dim=1280,
            prot_gcn_dims=[128, 256, 256],
            prot_fc_dims=[1024, 128],
            drug_node_in_dim=[66, 1], drug_node_h_dims=[128, 64],
            drug_edge_in_dim=[16, 1], drug_edge_h_dims=[32, 1],            
            drug_fc_dims=[1024, 128],
            mlp_dims=[1024, 512], mlp_dropout=0.25, 
            
            # node-level attention params
            attention_dim=128,num_heads=4, drug_dropout = 0.1, attention_dropout_rate=0.2,
            attn_dropout_rate=0.1,
            layer_norm_type='nn',
            sigmoid_scale_type='learnable',
            qk_norm=True,
            use_layer_scale=True,
            layer_scale_init=1e-5,
            norm_first=True,
            post_norm = True,
            scale_score = False,
            score_scaler=0.5,
            proj_dropout_rate=0.1,
            use_fusion = True,
            
            # drug self-attention params
            drug_self_attention = True,
            pair_num_heads=4, pair_dim=128, pair_attn_drop=0.1,
            # Drug positional embedding params
            num_kernel=64, max_pair_types=128*128, std_width=1.0, start=0.0, stop=9.0,
            
            # graph-level attention params
            use_graph_attention = True,
            graph_att_dropout = 0.1,
            base_dim=256,
            graph_att_gate_drop = 0.1,
            
            use_std=True): # Keep your original params
        super(Bidirection_Attention_Model, self).__init__()
        
        self.use_std = use_std

        
        self.use_fusion = use_fusion
        self.use_graph_attention = use_graph_attention
        self.drug_self_attention = drug_self_attention
        self.drug_model = DrugGVPModel(
            node_in_dim=drug_node_in_dim, node_h_dim=drug_node_h_dims,
            edge_in_dim=drug_edge_in_dim, edge_h_dim=drug_edge_h_dims,
            drop_rate = drug_dropout
        )

        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        
        drug_emb_dim = drug_node_h_dims[0]

        self.prot_model = Prot3DGraphModel(
            d_pretrained_emb=prot_emb_dim, d_gcn=prot_gcn_dims
        )
        if self.drug_self_attention:
            self.drug_positional_embedding = drug_gaussian_position_embedding(
                pair_dim=pair_dim, max_pair_types=max_pair_types, num_kernel=num_kernel,
                std_width=std_width, start=start, stop=stop
            )
            self.drug_self_attention = PairBiasedSelfAttention(
                d_model=drug_emb_dim, num_heads=pair_num_heads, pair_dim=pair_dim, attn_drop=pair_attn_drop
            )

        prot_emb_dim = prot_gcn_dims[-1]

        self.drug_fc = self.get_fc_layers(
            [drug_emb_dim] + drug_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

        self.prot_fc = self.get_fc_layers(
            [prot_emb_dim] + prot_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

        self.top_fc = self.get_fc_layers(
                    [drug_fc_dims[-1] + prot_fc_dims[-1]] + mlp_dims + [1],
                    dropout=mlp_dropout, batchnorm=False,
                    no_last_dropout=True, no_last_activation=True)
        print(f"top_fc layers: {[drug_fc_dims[-1] + prot_fc_dims[-1]] + mlp_dims + [1]}")
                

        self.protein_to_drug = SigmoidCrossAttention(query_dim =prot_gcn_dims[-1], key_dim=drug_fc_dims[-1], attention_dim=attention_dim, num_heads=num_heads,
                                                    attn_dropout_rate=attn_dropout_rate, layer_norm_type=layer_norm_type,
                                                    sigmoid_scale_type=sigmoid_scale_type,qk_norm=qk_norm,
                                                    use_layer_scale=use_layer_scale, layer_scale_init=layer_scale_init, norm_first=norm_first,
                                                    proj_dropout_rate=proj_dropout_rate, post_norm = post_norm, scale_score = scale_score,score_scaler=score_scaler)

        self.drug_to_protein = SigmoidCrossAttention(query_dim =drug_fc_dims[-1], key_dim=prot_gcn_dims[-1], attention_dim=attention_dim, num_heads=num_heads,
                                                    attn_dropout_rate=attn_dropout_rate, layer_norm_type=layer_norm_type,
                                                    sigmoid_scale_type=sigmoid_scale_type,qk_norm=qk_norm,
                                                    use_layer_scale=use_layer_scale, layer_scale_init=layer_scale_init, norm_first=norm_first,
                                                    proj_dropout_rate=proj_dropout_rate, post_norm = post_norm, scale_score = scale_score,score_scaler=score_scaler)
        if use_std:
            self.drug_pooling = MeanMaxStdPooling(
                    drug_emb_dim, 
                    use_std = False,
                    output_dim=drug_emb_dim  # Keep same dimension
                )
            self.prot_pooling = MeanMaxStdPooling(
                    prot_emb_dim, 
                    use_std = False,
                    output_dim=prot_emb_dim  # Keep same dimension
                )
        
        # These MLPs help combine original features with context features
        if self.use_fusion:
            # Use gated fusion instead of simple fusion
            self.fuse_drug = GatedFusion(drug_fc_dims[-1])
            self.fuse_prot = GatedFusion(prot_gcn_dims[-1])
        
        if self.use_graph_attention:
            self.d_p_attention = EnhancedBidirectionalCrossAttentionV2(drug_fc_dims[-1], prot_fc_dims[-1], base_dim, dropout=graph_att_dropout, gate_dropout=graph_att_gate_drop)
        
    def get_fc_layers(self, hidden_sizes,
            dropout=0, batchnorm=False,
            no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))
        return nn.Sequential(*layers)
    
    def forward(self, xd_input, xp_input):
        # --- 1. Get Initial Node Embeddings ---
        # Assume drug_model returns (graph_embedding, node_embeddings)
        # if not self.check_single_node_graphs(xd_input.batch, "drug_input"):
        #     return None  # Skip this batch
        # if not self.check_single_node_graphs(xp_input.batch, "protein_input"):
        #     return None  # Skip this batch

        _, h_d_nodes = self.drug_model(xd_input)

        batch_index = xd_input.batch
        
        h_d_dense, dense_mask = to_dense_batch(h_d_nodes, batch_index)
        
        if self.drug_self_attention:
            final_drug_PE_bias = self.drug_positional_embedding(xd_input)
            drug_self_attention_output, attn_weights = self.drug_self_attention(
                x=h_d_dense,
                pair_bias=final_drug_PE_bias,
                atom_mask=dense_mask
            )
        else:
            drug_self_attention_output = h_d_dense

        _, h_p_nodes = self.prot_model(xp_input)
        h_p_nodes_nodes, protein_mask = to_dense_batch(h_p_nodes, xp_input.batch)

        # --- 2. Perform Node-Level Conditioning ---
        protein_attention_output = self.protein_to_drug(
            query_feats=h_p_nodes_nodes,
            key_feats=drug_self_attention_output,
            query_mask=protein_mask,
            key_mask=dense_mask
        )

        drug_attention_output = self.drug_to_protein(
            query_feats=drug_self_attention_output,
            key_feats=h_p_nodes_nodes,
            query_mask=dense_mask,
            key_mask=protein_mask
        )
        
        # convert back to sparse
        drug_attention_sparse = drug_attention_output.view(-1, h_d_nodes.size(1))[dense_mask.view(-1)]
        # drug_attention_sparse = drug_attention_output[dense_mask]
        protein_attention_sparse = protein_attention_output.view(-1, h_p_nodes.size(1))[protein_mask.view(-1)]

        if self.use_fusion:
            # Gated fusion of original and context
            h_d_fused = self.fuse_drug(h_d_nodes, drug_attention_sparse)
            h_p_fused = self.fuse_prot(h_p_nodes, protein_attention_sparse)
        else:
            # Direct use of context (rely on internal residuals)
            h_d_fused = drug_attention_sparse
            h_p_fused = protein_attention_sparse
            

        if self.use_std:
            # --- 3. Graph-Level Pooling with MeanMaxStd ---
            zd_conditioned = self.drug_pooling(h_d_fused, xd_input.batch)
            zp_conditioned = self.prot_pooling(h_p_fused, xp_input.batch)
            # print("drug dimension after pooling:", zd_conditioned.size())
        else:
            zd_conditioned = global_mean_pool(h_d_fused, xd_input.batch)
            zp_conditioned = global_mean_pool(h_p_fused, xp_input.batch)
        # --- 5. Final Prediction ---
        xd = self.drug_fc(zd_conditioned)
        xp = self.prot_fc(zp_conditioned)
        if self.use_graph_attention:
            # Graph-level bidirectional attention
            final_dp = self.d_p_attention(xd, xp)
        else:
            # Simple concatenation
            final_dp = torch.cat([xd, xp], dim=-1)
        final_dp = self.attention_dropout(final_dp)
        yh = self.top_fc(final_dp)
        return yh
