# attention.py
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch_geometric.utils import to_dense_batch

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
        Root Mean Square Layer Normalization.
        Source: https://arxiv.org/abs/1910.07467
        adapted from:https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
        Root Mean Square Layer Normalization
            :param d: model size
            :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
            :param eps:  epsilon value, default 1e-8
            :param bias: whether use bias term for RMSNorm, disabled by
                default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)
    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            d_x = partial_size
        
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class SigmoidCrossAttention(nn.Module):
    """
    Generic Sigmoid Cross-Attention Module
    
    Can be used for any cross-modal attention:
    - query_dim: dimension of the querying entity (who is asking)
    - key_dim: dimension of the key/value entity (who is being attended to)
    
    The output will have the same dimension as the query entity.
    """
    
    def __init__(
        self,
        query_dim=128,      # Dimension of querying entity (e.g., protein when protein→drug)
        key_dim=128,        # Dimension of key/value entity (e.g., drug when protein→drug)
        attention_dim=128,  # Internal attention dimension
        num_heads=4,
        attn_dropout_rate=0.1,
        proj_dropout_rate=0.1,
        layer_norm_type='nn',
        sigmoid_scale_type='fixed',
        qk_norm=True,
        use_layer_scale=True,
        layer_scale_init=1e-5,
        norm_first=True,
        post_norm=True,
        scale_score=False,  # For sigmoid, usually False
        score_scaler=0.5,
        default_bias=-3.0,
        device='cuda'
    ):
        super().__init__()
        assert attention_dim % num_heads == 0, f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
        
        # Dimensions
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        # Configuration flags
        self.norm_first = norm_first
        self.scale_score = scale_score
        self.qk_norm = qk_norm
        self.post_norm = post_norm
        self.score_scaler = score_scaler
        
        # Sigmoid scaling
        if sigmoid_scale_type == 'learnable':
            self.sigmoid_scale = nn.Parameter(torch.ones(1) * (self.head_dim ** -0.25))
        elif sigmoid_scale_type == 'fixed':
            self.register_buffer('sigmoid_scale', torch.tensor(self.head_dim ** -0.25, dtype=torch.float32))
        else:
            self.register_buffer('sigmoid_scale', torch.tensor(1.0, dtype=torch.float32))
        
        # Q, K, V projections
        self.q_proj = nn.Linear(query_dim, attention_dim, bias=False)  # Query from querying entity
        self.k_proj = nn.Linear(key_dim, attention_dim, bias=False)    # Key from key entity
        self.v_proj = nn.Linear(key_dim, attention_dim, bias=False)    # Value from key entity
        self.out_proj = nn.Linear(attention_dim, query_dim, bias=False)  # Output back to query dimension
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.proj_dropout = nn.Dropout(proj_dropout_rate)
        
        # Normalization layers
        norm_class = RMSNorm if layer_norm_type.lower() == 'rms_norm' else nn.LayerNorm
        
        # QK normalization
        if self.qk_norm:
            self.q_norm = norm_class(self.head_dim)
            self.k_norm = norm_class(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        
        # Input normalization for query features
        self.query_norm = norm_class(query_dim)
        
        # Output normalization
        if self.post_norm:
            self.out_norm = norm_class(query_dim)
        
        # LayerScale for training stability
        if use_layer_scale:
            self.gamma = nn.Parameter(torch.ones(query_dim) * layer_scale_init)
        else:
            self.gamma = None
        
        # Default sigmoid bias
        self.register_buffer('default_sigmoid_bias', torch.tensor(default_bias, dtype=torch.float32))
        
        self.to(device)
    
    def hardware_optimized_sigmoid(self, x):
        """
        Hardware-optimized sigmoid implementation: 0.5 * (1 + tanh(0.5 * x))
        More efficient on modern GPUs than torch.sigmoid
        """
        return 0.5 * (1.0 + torch.tanh(0.5 * x))
    
    def create_cross_attention_mask(self, query_mask, key_mask):
        """
        Build a broadcastable mask for cross-attention scores
        
        Args:
            query_mask: [B, Nq] (True = valid query position)
            key_mask: [B, Nk] (True = valid key position)
        Returns:
            attn_mask: [B, 1, Nq, Nk] (True = valid interaction)
        """
        # Create cross-product of masks
        attn_mask = query_mask.unsqueeze(-1) & key_mask.unsqueeze(1)  # [B, Nq, Nk]
        return attn_mask.unsqueeze(1)  # [B, 1, Nq, Nk] for broadcasting with heads
    
    def forward(self, query_feats, key_feats, query_mask=None, key_mask=None, return_attention=False):
        """
        Forward pass for sigmoid cross-attention
        
        Args:
            query_feats: [B, Nq, query_dim] - Features from querying entity
            key_feats: [B, Nk, key_dim] - Features from key/value entity
            query_mask: [B, Nq] - Valid positions in query (optional)
            key_mask: [B, Nk] - Valid positions in key (optional)
            return_attention: Whether to return attention weights
        
        Returns:
            output: [B, Nq, query_dim] - Query features enhanced with key information
            attn_weights: [B, num_heads, Nq, Nk] - Attention weights (if return_attention=True)
        """
        # Store residual
        residual = query_feats
        
        # Pre-normalization if specified
        if self.norm_first:
            query_feats = self.query_norm(query_feats)
        
        # Get batch dimensions
        B, Nq, _ = query_feats.shape
        B, Nk, _ = key_feats.shape
        
        # Project to Q, K, V
        q = self.q_proj(query_feats)  # [B, Nq, attention_dim]
        k = self.k_proj(key_feats)     # [B, Nk, attention_dim]
        v = self.v_proj(key_feats)     # [B, Nk, attention_dim]
        
        # Reshape for multi-head attention
        # [B, seq_len, attention_dim] -> [B, num_heads, seq_len, head_dim]
        q = q.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)  # Normalize over head_dim
            k = self.k_norm(k)  # Normalize over head_dim
        
        # Apply sigmoid scaling (NOT standard softmax scaling)
        if not self.scale_score:
            q = q * self.sigmoid_scale
            k = k * self.sigmoid_scale
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # [B, num_heads, Nq, Nk]
        
        # Optional: apply standard scaling (usually not used with sigmoid)
        if self.scale_score:
            scores = scores / (self.head_dim ** self.score_scaler)
        
        # Apply masking if provided
        if query_mask is not None and key_mask is not None:
            attn_mask = self.create_cross_attention_mask(query_mask, key_mask)
            scores = scores.masked_fill(~attn_mask, -1e9)
        elif key_mask is not None:
            # Only mask invalid keys (more common case)
            key_mask_expanded = key_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, Nk]
            scores = scores.masked_fill(~key_mask_expanded, -1e9)
        
        # Compute dynamic sigmoid bias based on number of keys
        if key_mask is not None:
            # Count valid keys per batch
            num_valid_keys = key_mask.sum(-1).clamp_min(1).to(scores.dtype)  # [B]
            bias = -torch.log(num_valid_keys)  # [B]
            bias = bias.view(B, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting
        else:
            bias = self.default_sigmoid_bias.view(1, 1, 1, 1)
        
        # Add bias to scores
        scores = scores + bias
        
        # Apply sigmoid activation
        attn_weights = self.hardware_optimized_sigmoid(scores)  # [B, num_heads, Nq, Nk]
        
        # Apply attention dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [B, num_heads, Nq, head_dim]
        
        # Reshape back to [B, Nq, attention_dim]
        context = context.permute(0, 2, 1, 3).contiguous()  # [B, Nq, num_heads, head_dim]
        context = context.view(B, Nq, self.attention_dim)    # [B, Nq, attention_dim]
        
        # Output projection
        output = self.out_proj(context)  # [B, Nq, query_dim]
        output = self.proj_dropout(output)
        
        # Apply LayerScale if enabled
        if self.gamma is not None:
            output = self.gamma * output
        
        # Residual connection
        output = residual + output
        
        # Post-normalization if specified
        if self.post_norm:
            output = self.out_norm(output)
        
        # Zero out padded positions in output
        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1).float()
        
        if return_attention:
            return output, attn_weights
        return output
    

################bidirectional prediction model


class EnhancedBidirectionalCrossAttentionV2(nn.Module):
    """
    Graph-level bidirectional sigmoid gating (no cross-batch leakage).
    Upgrades:
      - LayerNorm on Q/K
      - 1/sqrt(d) scaling + positive temperature (softplus)
      - Per-channel gates (vector), not a single scalar
      - Gate dropout
      - Correct residual (Identity)
      - Layerscale
    """
    def __init__(self, drug_dim, prot_dim, base_dim, dropout=0.1, gate_dropout=0.1):
        super().__init__()
        self.base_dim = base_dim
        self.scale = base_dim ** -0.5

        # Drug -> Protein projections
        self.drug_q = nn.Linear(drug_dim, base_dim)
        self.prot_k = nn.Linear(prot_dim, base_dim)
        self.prot_v = nn.Linear(prot_dim, base_dim)

        # Protein -> Drug projections
        self.prot_q = nn.Linear(prot_dim, base_dim)
        self.drug_k = nn.Linear(drug_dim, base_dim)
        self.drug_v = nn.Linear(drug_dim, base_dim)

        # Q/K norm (shared across channels)
        # Separate norms for each direction
        self.drug_q_norm = nn.LayerNorm(base_dim)
        self.prot_q_norm = nn.LayerNorm(base_dim)
        self.drug_k_norm = nn.LayerNorm(base_dim)
        self.prot_k_norm = nn.LayerNorm(base_dim)

        # Temperature (positive)
        self._log_tau = nn.Parameter(torch.zeros(1))  # tau = softplus(_log_tau) + eps

        # Optional learned biases on gate logits (one per direction, per channel)
        self.bias_dp = nn.Parameter(torch.zeros(base_dim))  # drug<-prot gate bias
        self.bias_pd = nn.Parameter(torch.zeros(base_dim))  # prot<-drug gate bias

        # Gate dropout
        self.gate_dropout = nn.Dropout(gate_dropout)

        # Final projection (concat [drug, drug_ctx, prot, prot_ctx] -> [drug+prot])
        combined_in = (drug_dim + base_dim) + (prot_dim + base_dim)
        combined_out = drug_dim + prot_dim
        self.final_proj = nn.Sequential(
            nn.Linear(combined_in, combined_out),
            nn.ReLU(),  # mild nonlinearity helps
            nn.Dropout(dropout)
        )

        # Residual + norm (dims match → Identity on residual)
        self.residual_proj = nn.Identity()
        self.norm = nn.LayerNorm(combined_out)

        # Tiny layerscale on the update
        self.layerscale = nn.Parameter(torch.ones(1) * 1e-2)

    def forward(self, drug_feat, prot_feat):
        """
        drug_feat: [B, drug_dim]
        prot_feat: [B, prot_dim]
        Returns:   [B, drug_dim + prot_dim]
        """
        B = drug_feat.size(0)
        original = torch.cat([drug_feat, prot_feat], dim=-1)  # [B, drug+prot]

        # --- Drug attends to Protein (per-sample, per-channel gate) ---
        q_d = self.drug_q_norm(self.drug_q(drug_feat))  # [B, D]
        k_p = self.prot_k_norm(self.prot_k(prot_feat))  # [B, D]
        v_p = self.prot_v(prot_feat)               # [B, D]

        tau = F.softplus(self._log_tau) + 1e-6     # positive temperature
        logit_dp = (q_d * k_p) * (self.scale / tau) + self.bias_dp  # [B, D]
        gate_dp = torch.sigmoid(logit_dp)                              # [B, D]
        gate_dp = self.gate_dropout(gate_dp)                           # [B, D]
        drug_context = gate_dp * v_p                                   # [B, D]

        # --- Protein attends to Drug (per-sample, per-channel gate) ---
        q_p = self.prot_q_norm(self.prot_q(prot_feat))  # [B, D]
        k_d = self.drug_k_norm(self.drug_k(drug_feat))  # [B, D]
        v_d = self.drug_v(drug_feat)               # [B, D]

        logit_pd = (q_p * k_d) * (self.scale / tau) + self.bias_pd  # [B, D]
        gate_pd = torch.sigmoid(logit_pd)                            # [B, D]
        gate_pd = self.gate_dropout(gate_pd)
        prot_context = gate_pd * v_d                                 # [B, D]

        # --- Combine & project ---
        enhanced_drug = torch.cat([drug_feat, drug_context], dim=-1)  # [B, drug+base]
        enhanced_prot = torch.cat([prot_feat, prot_context], dim=-1)  # [B, prot+base]
        combined = torch.cat([enhanced_drug, enhanced_prot], dim=-1)  # [B, drug+base+prot+base]

        update = self.final_proj(combined)                            # [B, drug+prot]

        # Residual + layerscale + norm
        out = self.residual_proj(original) + self.layerscale * update
        out = self.norm(out)
        return out


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(dim * 2, dim)
    
    def forward(self, original, context):
        combined = torch.cat([original, context], dim=-1)
        gate = self.gate(combined)
        
        transformed = self.transform(combined)
        return gate * transformed + (1 - gate) * original
    
    
    

class PairBiasedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pair_dim: int, attn_drop: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.h = num_heads
        self.dh = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # pair_dim -> per-head bias (like Uni-Mol's pair2head projection)
        self.pair2head = nn.Linear(pair_dim, num_heads, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
    
    def forward(self, x, pair_bias, atom_mask=None):
        """
        Uni-Mol style drug self-attention with position encoding integration
        
        Args:
            x: [B, N, D] - drug node features
            pair_bias: [B, N, N, pair_dim] - pairwise position encoding (from SE3InvariantKernel equivalent)
            atom_mask: [B, N] - atom validity mask (1 for valid, 0 for padding)
            
        Returns:
            final_out: [B, N, D] - enhanced drug features
            attn_weights: [B, H, N, N] - attention weights
        """
        B, N, D = x.shape
        
        # 1. Multi-head projections (standard transformer)
        q = self.q_proj(x).view(B, N, self.h, self.dh).transpose(1, 2)  # [B, H, N, dh]
        k = self.k_proj(x).view(B, N, self.h, self.dh).transpose(1, 2)  # [B, H, N, dh]
        v = self.v_proj(x).view(B, N, self.h, self.dh).transpose(1, 2)  # [B, H, N, dh]
        
        # 2. Compute base attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)  # [B, H, N, N]
        
        # 3. Integrate position encoding (Uni-Mol style)
        # Project pair features to per-head bias
        pair_bias_projected = self.pair2head(pair_bias)  # [B, N, N, H]
        pair_bias_projected = pair_bias_projected.permute(0, 3, 1, 2).contiguous()  # [B, H, N, N]
        
        # 4. Clean pair bias (remove padding contamination)
        if atom_mask is not None:
            pair_mask = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)  # [B, N, N]
            pair_bias_projected = pair_bias_projected.masked_fill(~pair_mask.unsqueeze(1), 0.0)
        
        # 5. Add position encoding to attention scores (core Uni-Mol approach)
        scores = scores + pair_bias_projected  # [B, H, N, N]
        
        # 6. Apply key masking (prevent attending to padding)
        if atom_mask is not None:
            key_mask = ~atom_mask  # [B, N] - True for padding positions
            neg = torch.full((), -1e6, dtype=scores.dtype, device=scores.device)
            scores = scores.masked_fill(key_mask[:, None, None, :], neg)
        
        # 7. Compute attention weights (no NaN generation)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, N, N]
        attn_weights = self.attn_drop(attn_weights)
        
        # 8. Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, N, dh]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        final_out = self.o_proj(out)
        
        # 9. Final cleanup (zero padding outputs)
        if atom_mask is not None:
            final_out = final_out.masked_fill(~atom_mask.unsqueeze(-1), 0.0)
        
        return final_out, attn_weights
    



def compute_distances_official_style(pos):
    """
    Distance computation following official Uni-Mol style
    
    The official code: delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
    But this assumes pos is [batch, N, 3], not [N, 3]!
    """
    
    # Check input shape
    if pos.dim() == 2:  # [N, 3] - your case
        print(f"Input shape: {pos.shape} (single molecule)")
        
        # Method 1: Official style adapted for 2D input
        delta_pos = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, 3]
        dist = delta_pos.norm(dim=-1)  # [N, N]
        
    elif pos.dim() == 3:  # [batch, N, 3] - official assumption
        # print(f"Input shape: {pos.shape} (batched molecules)")
        
        # Method 2: Official style for 3D input
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)  # [batch, N, N, 3]  
        dist = delta_pos.norm(dim=-1)  # [batch, N, N]
    
    else:
        raise ValueError(f"Unexpected pos shape: {pos.shape}")
    
    return dist


class Embedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
    ):
        super(Embedding, self).__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self._normal_init()

        if padding_idx is not None:
            self.weight.data[self.padding_idx].zero_()

    def _normal_init(self, std=0.02):
        nn.init.normal_(self.weight, mean=0.0, std=std)


    
def gaussian(x, mean, std):
    """Official Uni-Mol Gaussian function (JIT compiled for speed)"""
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianKernel(nn.Module):
    """Official Uni-Mol GaussianKernel implementation"""
    
    def __init__(self, K=128, num_pair=512, std_width=1.0, start=0.0, stop=9.0):
        super().__init__()
        self.K = K
        
        # Fixed Gaussian parameters
        mean = torch.linspace(start, stop, K)
        std = std_width * (mean[1] - mean[0])
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
        # Learnable pair-type parameters (embeddings)
        self.mul = Embedding(num_pair, 1, padding_idx=0)
        self.bias = Embedding(num_pair, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1.0)

    def forward(self, x, atom_pair):
        """
        Args:
            x: distance tensor [batch_size, num_atoms, num_atoms] 
            atom_pair: pair type indices [batch_size, num_atoms, num_atoms]
        Returns:
            Gaussian features [batch_size, num_atoms, num_atoms, K]
        """
        mul = self.mul(atom_pair).abs().sum(dim=-2)  # [batch, num_atoms, num_atoms]
        bias = self.bias(atom_pair).sum(dim=-2)      # [batch, num_atoms, num_atoms]
        x = mul * x.unsqueeze(-1) + bias             # [batch, num_atoms, num_atoms, 1]
        x = x.expand(-1, -1, -1, self.K)             # [batch, num_atoms, num_atoms, K]
        mean = self.mean.float().view(-1)
        return gaussian(x.float(), mean, self.std)

    
class drug_gaussian_position_embedding(nn.Module):
    def __init__(self, pair_dim=128,max_pair_types = 128 * 128, num_kernel=64, 
                    std_width=1.0, start=0.0, stop=9.0):
        super(drug_gaussian_position_embedding, self).__init__()
        self.gaussian_k = GaussianKernel(
            K=num_kernel,
            num_pair=max_pair_types,
            std_width=std_width,
            start=start,
            stop=stop,
            )
        self.out_proj = nn.Sequential(
                nn.Linear(num_kernel, pair_dim),
                nn.ReLU()
            )
    def forward(self, xd_input):
        """
        xd_input: Drug graph data object with fields:
            - x: atom coordinates [N, 3]
            - final_pair_type: pair type indices [N, N]
            - batch: batch indices [N]
        Returns:
            final_drug_PE_bias: Position embedding with bias [N, pair_dim]
        """
        coords = xd_input.x
        atomic_numbers = xd_input.final_pair_type
        batch_index = xd_input.batch 
        coords_batch, atom_mask = to_dense_batch(coords, batch_index)
        atomic_numbers_batch = atomic_numbers
        
        distances_correct = compute_distances_official_style(coords_batch)
        
        final_drug_PE = self.gaussian_k(distances_correct, atomic_numbers_batch)
        final_drug_PE_bias = self.out_proj(final_drug_PE)
        
        return final_drug_PE_bias