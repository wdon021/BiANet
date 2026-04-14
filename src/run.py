# run.py
import argparse
import pickle
from parsing import add_train_args
from utils import set_global_seed
from pathlib import Path
import os
from train import DTAExperiment


parser = argparse.ArgumentParser(description='Run BiANet experiment')
add_train_args(parser)
args = parser.parse_args([])
args.n_epochs = 300
args.n_ensembles = 3
args.patience = 70
args.uncertainty = True
args.recalibrate = True
args.save_prediction =True
args.save_log = True
args.save_checkpoint = True
args.seed = 55
args.monitor_metric = 'spearman'
args.base_path = 'C:/'
args.output_dir = 'DTI Project/baseModel/Sigmoid_cross_attention/'
trial_number =12


davis_splits = pickle.load(open(os.path.join(args.base_path, 'davis_splits.pkl'), 'rb'))
data_splits, df_splits = davis_splits


# Hyperparameters to tune for the Drug VGIB baseline
set_global_seed(args.seed + trial_number)
lr = 0.00011963059243916732
weight_decay = 1.0698022288580466e-05
batch_size = 128
base_dim = 64
drug_dropout =0.2
attention_dropout = 0.4
attention_dim_model = 64
graph_att_dropout = 0.2
graph_att_gate_drop =0.2
num_heads = 1
mlp_dropout = 0.3
mlp_dims =[1024, 512]
use_std = True

# ========== NEW HYPERPARAMETERS TO TUNE ==========

# Node-level attention architectural choices
layer_norm_type = 'nn'
sigmoid_scale_type = 'fixed'
qk_norm = True
use_layer_scale = False

# Layer scale initialization (only if use_layer_scale is True)
if use_layer_scale:
    layer_scale_init =4.6882017010465915e-06
else:
    layer_scale_init = 1e-5  # Default value

# Normalization strategies
norm_first = True
post_norm =True

# Projection dropout (separate from attention dropout)
proj_dropout_rate =0.0

# Feature fusion strategy
use_fusion = True

# Drug self-attention parameters
drug_self_attention = False

if drug_self_attention:
    pair_num_heads = 2
    pair_dim = 128
    pair_attn_drop = 0.0
else:
    pair_num_heads = 4  # Default values
    pair_dim = 128
    pair_attn_drop = 0.1

# Graph-level attention
use_graph_attention = False

# ========== FIXED PARAMETERS (rarely need tuning) ==========
# These are typically optimal at their default values
scale_score = False  # Usually False for sigmoid
score_scaler = 1.0  # Only matters if scale_score is True
attn_dropout_rate = attention_dropout  # Use same as attention_dropout to avoid redundancy

current_trial_specific_output_dir = os.path.join(args.output_dir, f"testing/trial_{trial_number}")
Path(current_trial_specific_output_dir).mkdir(parents=True, exist_ok=True)
# For tuning, reduce epochs and ensembles to speed up the search

# For tuning, reduce epochs and ensembles to speed up the search
    
exp = DTAExperiment(
            
        prot_gcn_dims=args.prot_gcn_dims,
        prot_fc_dims=args.prot_fc_dims,
        drug_gcn_dims=args.drug_gcn_dims,
        drug_fc_dims=args.drug_fc_dims,
        mlp_dims=mlp_dims,
        mlp_dropout=mlp_dropout,
        
        # Training parameters
        n_ensembles=args.n_ensembles,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=args.seed + trial_number,
        
        # Experiment settings
        uncertainty=args.uncertainty,
        parallel=args.parallel,
        output_dir=current_trial_specific_output_dir,
        save_log=args.save_log,
        precomputed_data_splits=data_splits,
        precomputed_df_splits=df_splits,
        
        # Attention dimensions
        base_dim=base_dim,
        attention_dim=attention_dim_model,
        num_heads=num_heads,
        
        # Dropout rates
        drug_dropout=drug_dropout,
        attn_dropout_rate=attn_dropout_rate,
        proj_dropout_rate=proj_dropout_rate,
        
        # Node-level attention configuration
        layer_norm_type=layer_norm_type,
        sigmoid_scale_type=sigmoid_scale_type,
        qk_norm=qk_norm,
        use_layer_scale=use_layer_scale,
        layer_scale_init=layer_scale_init,
        norm_first=norm_first,
        post_norm=post_norm,
        scale_score=scale_score,
        score_scaler=score_scaler,
        use_fusion=use_fusion,
        
        # Drug self-attention
        drug_self_attention=drug_self_attention,
        pair_num_heads=pair_num_heads,
        pair_dim=pair_dim,
        pair_attn_drop=pair_attn_drop,
        
        # Graph-level attention
        use_graph_attention=use_graph_attention,
        graph_att_dropout=graph_att_dropout,
        graph_att_gate_drop=graph_att_gate_drop,
        use_std=use_std
        )



if args.save_prediction or args.save_log or args.save_checkpoint:
    exp.saver.save_config(args.__dict__, 'args.yaml')
# args.n_epochs = 1
exp.train(n_epochs=args.n_epochs, patience=args.patience,
    eval_freq=args.eval_freq, test_freq=args.test_freq,
    monitoring_score=args.monitor_metric)

if args.recalibrate:
    val_results = exp.test(test_loader=exp.task_loader['valid'], test_df=exp.task_df['valid'],
        test_tag="valid set", print_log=True)
recalib_df = val_results['df'] if args.recalibrate else None

test_results = exp.test(save_prediction=args.save_prediction, recalib_df=recalib_df,
    test_tag="Bidirectional davis S3 Ensemble spearman sigmoid prot feature model", print_log=True)
