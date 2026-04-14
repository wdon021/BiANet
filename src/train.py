# train.py
from utils import(
    Logger,
    Saver,
    EarlyStopping
)
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from metrics import evaluation_metrics
from model import Bidirection_Attention_Model
from joblib import Parallel, delayed
import math
import torch.nn.functional as F
import uncertainty_toolbox as uct

def _parallel_train_per_epoch(
        kwargs=None, test_loader=None,
        n_epochs=None, eval_freq=None, test_freq=None,
        monitoring_score='pearson',
        loss_fn=None, logger=None,
        test_after_train=True, mode = 'train'
    ):

        global_step = 0
        midx = kwargs['midx']
        model = kwargs['model']
        optimizer = kwargs['optimizer']
        train_loader = kwargs['train_loader']
        valid_loader = kwargs['valid_loader']
        scheduler = kwargs['scheduler']
        device = kwargs['device']
        stopper = kwargs['stopper']
        best_model_state_dict = kwargs['best_model_state_dict']
        if stopper.early_stop:
            return kwargs

        model.train()
        for epoch in range(1, n_epochs + 1):
            total_loss = 0
            for step, batch in enumerate(train_loader, start=1):
                xd = batch['drug'].to(device)
                xp = batch['protein'].to(device)
                xd.protein = xp.name
                y = batch['y'].to(device)
                optimizer.zero_grad()
                yh= model(xd, xp)
                loss = loss_fn(yh, y.view(-1, 1))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                scheduler.step()
                global_step += 1
                # loss = first_loss + mi_loss # trade off between aggressively compress the information (mi_weight=1),prediction using the entire graph structure (mi_weight=0) 
                pred_loss = loss
                total_loss += loss.item()
                if step % 50 == 0:
                    print(f"...epoch:{epoch} pred_loss: {pred_loss.item()}...")

                last_pred_loss = pred_loss.item()


            train_loss = total_loss / step
            if epoch % eval_freq == 0:
                val_results = _parallel_test(
                    {'model': model, 'midx': midx, 'test_loader': valid_loader, 'device': device},
                    loss_fn=loss_fn, logger=logger, mode = 'test'
                )
                is_best = stopper.update(val_results['metrics'][monitoring_score])
                if is_best:
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                logger.info(f"M-{midx} E-{epoch} | Train Loss: {train_loss:.4f} | Predict Loss :{last_pred_loss:.4f} | Valid Loss: {val_results['loss']:.4f} | "\
                    + ' | '.join([f'{k}: {v:.4f}' for k, v in val_results['metrics'].items()])
                    + f" | best {monitoring_score}: {stopper.best_score:.4f}"
                    )
            if test_freq is not None and epoch % test_freq == 0:
                test_results = _parallel_test(
                    {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
                    loss_fn=loss_fn, logger=logger,mode = 'test'
                )
                logger.info(f"M-{midx} E-{epoch} | Test Loss: {test_results['loss']:.4f} | "\
                    + ' | '.join([f'{k}: {v:.4f}' for k, v in test_results['metrics'].items()])
                    )

            if stopper.early_stop:
                logger.info('Eearly stop at epoch {}'.format(epoch))
                break

        if best_model_state_dict is not None:
            model.load_state_dict(best_model_state_dict)
        if test_after_train:
            test_results = _parallel_test(
                {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
                loss_fn=loss_fn,
                test_tag=f"Model {midx}", print_log=True, logger=logger,mode = 'test'
            )
        rets = dict(midx = midx, model = model)
        return rets


def _parallel_test(
    kwargs=None, loss_fn=None,
    test_tag=None, print_log=False, logger=None,mode=None
):
    midx = kwargs['midx']
    model = kwargs['model']
    test_loader = kwargs['test_loader']
    device = kwargs['device']
    model.eval()
    yt, yp, total_loss = torch.Tensor(), torch.Tensor(), 0
    with torch.no_grad():
        for step, batch in enumerate(test_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            xd.protein = xp.name
            y = batch['y'].to(device)
            yh = model(xd, xp)
            loss = loss_fn(yh, y.view(-1, 1))
            total_loss += loss.item()
            yp = torch.cat([yp, yh.detach().cpu()], dim=0)
            yt = torch.cat([yt, y.detach().cpu()], dim=0)
    yt = yt.numpy()
    yp = yp.view(-1).numpy()
    results = {
        'midx': midx,
        'y_true': yt,
        'y_pred': yp,
        'loss': total_loss / step,
    }
    eval_metrics = evaluation_metrics(
        yt, yp,
        eval_metrics=['mse', 'spearman', 'pearson']
    )
    results['metrics'] = eval_metrics
    if print_log:
        logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
            + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
    return results




def pad_2d_feat(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    assert len(samples[0].shape) == 3
    feat_size = samples[0].shape[-1]
    tensor = torch.full(
        [batch_size, pad_len, pad_len, feat_size], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0], : samples[i].shape[1]] = samples[i]
    return tensor

def exclude_keys_collater(batch):
    from torch_geometric.data import Batch
    import torch
    
    drug_graphs = [item['drug'] for item in batch]
    protein_graphs = [item['protein'] for item in batch]
    y_values = [item['y'] for item in batch]
    
    # Extract final_pair_type (don't delete it)
    final_pair_type_data = None
    if hasattr(drug_graphs[0], 'final_pair_type'):
        final_pair_types = [drug.final_pair_type for drug in drug_graphs]
        max_atoms = max([fpt.shape[0] for fpt in final_pair_types])
        # max_atoms = (max_atoms + 1 + 3) // 4 * 4 - 1
        final_pair_type_data = pad_2d_feat(final_pair_types, max_atoms)
    
    # Use exclude_keys to ignore problematic attributes during batching
    batched_drugs = Batch.from_data_list(drug_graphs, exclude_keys=['final_pair_type'])
    batched_proteins = Batch.from_data_list(protein_graphs)
    batched_y = torch.tensor(y_values, dtype=torch.float32)
    
    # Add back final_pair_type
    if final_pair_type_data is not None:
        batched_drugs.final_pair_type = final_pair_type_data
    
    return {'drug': batched_drugs, 'protein': batched_proteins, 'y': batched_y}



def _build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or "norm" in n.lower() or "layernorm" in n.lower() or "bn" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999), eps=1e-8
    )
    
def _build_scheduler(optimizer, steps_per_epoch, n_epochs, warmup_ratio=0.05, min_lr_ratio=1e-2):
    total_steps = max(1, steps_per_epoch * n_epochs)
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    base_lr = optimizer.param_groups[0]["lr"]
    eta_min = base_lr * min_lr_ratio

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step, 1) / float(warmup_steps)  # linear 0→1
        t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return eta_min / base_lr + 0.5 * (1.0 - eta_min / base_lr) * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



class DTAExperiment(object):
    def __init__(self,prot_gcn_dims=[128, 128, 128], prot_gcn_bn=False,
        prot_fc_dims=[1024, 128],
        drug_in_dim=66, drug_fc_dims=[1024, 128], drug_gcn_dims=[128, 64],
        mlp_dims=[1024, 512], mlp_dropout=0.25,
        n_ensembles=8, n_epochs=500, batch_size=256,
        lr=0.001,
        weight_decay=0.0001,
        seed=42, onthefly=False,
        uncertainty=False, parallel=False,
        output_dir='Training_output/', save_log=False,
        precomputed_data_splits=None, # Dict {'train': DTA_dataset, ...}
        precomputed_df_splits=None,
        
        # node-level attention params
        attention_dim= 128,num_heads=4, drug_dropout = 0.2, attention_dropout_rate=0.2,
        attn_dropout_rate= 0.2,
        layer_norm_type='nn',
        sigmoid_scale_type='fixed',
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
        # num_kernel=64, max_pair_types=128*128, std_width=1.0, start=0.0, stop=9.0,
        
        
        # graph-level attention params
        use_graph_attention = True,
        graph_att_dropout = 0.1,
        base_dim=256,
        graph_att_gate_drop = 0.1,
        use_std = True
    ):
        self.saver = Saver(output_dir)
        self.logger = Logger(logfile=self.saver.save_dir/'exp.log' if save_log else None)
        self.num_heads = num_heads 
        self.uncertainty = uncertainty
        self.parallel = parallel
        self.n_ensembles = n_ensembles
        if self.uncertainty and self.n_ensembles < 2:
            raise ValueError('n_ensembles must be greater than 1 when uncertainty is True')
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self._task_loader = None # Cache for loaders
        n_gpus = torch.cuda.device_count()
        if self.parallel and n_gpus < self.n_ensembles:
            self.logger.warning(f"Visible GPUs ({n_gpus}) is fewer than "
            f"number of models ({self.n_ensembles}). Some models will be run on the same GPU"
            )
        self.devices = [torch.device(f'cuda:{i % n_gpus}')
            for i in range(self.n_ensembles)]

        self.model_config = dict(
            prot_emb_dim=1280,
            prot_gcn_dims=prot_gcn_dims,
            prot_fc_dims=prot_fc_dims,
            drug_node_h_dims=drug_gcn_dims,  
            drug_node_in_dim=[66, 1],         
            drug_fc_dims=drug_fc_dims,
            mlp_dims=mlp_dims, mlp_dropout=mlp_dropout,
            
            # node-level attention params
            attention_dim=attention_dim,num_heads=num_heads, drug_dropout = drug_dropout, attention_dropout_rate=attention_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            layer_norm_type=layer_norm_type,
            sigmoid_scale_type=sigmoid_scale_type,
            qk_norm=qk_norm,
            use_layer_scale=use_layer_scale,
            layer_scale_init=layer_scale_init,
            norm_first=norm_first,
            post_norm = post_norm,
            scale_score = scale_score,
            score_scaler=score_scaler,
            proj_dropout_rate=proj_dropout_rate,
            use_fusion = use_fusion,

            # drug self-attention params
            drug_self_attention = drug_self_attention,
            pair_num_heads=pair_num_heads, pair_dim=pair_dim, pair_attn_drop=pair_attn_drop,
            # Drug positional embedding params
            # num_kernel=num_kernel, max_pair_types=max_pair_types, std_width=std_width, start=start, stop=stop,

            # graph-level att ention params
            use_graph_attention = use_graph_attention,
            graph_att_dropout = graph_att_dropout,
            base_dim = base_dim,
            graph_att_gate_drop = graph_att_gate_drop,
            use_std = use_std
        )

        
        self.criterion = F.mse_loss


        self._data_splits = precomputed_data_splits # Store in _data_splits
        self._df_splits = precomputed_df_splits   # Store in _df_splits
        self.build_model()
        self.logger.info(self.models[0])
        self.logger.info(self.optimizers[0])
    def build_model(self):
        self.models = [Bidirection_Attention_Model(**self.model_config).to(self.devices[i])
                        for i in range(self.n_ensembles)]
        
        self.optimizers, self.schedulers = [], []
        steps_per_epoch = len(self.task_loader['train'])  # batches per epoch
        for i, m in enumerate(self.models):
            opt = _build_optimizer(m, lr=self.lr, weight_decay=self.weight_decay)
            sch = _build_scheduler(opt, steps_per_epoch, self.n_epochs, warmup_ratio=0.05, min_lr_ratio=1e-2)
            self.optimizers.append(opt)
            self.schedulers.append(sch)
            
    def _get_data_loader(self, dataset, shuffle=False):
        return DataLoader(  # ← FIXED: Use standard PyTorch DataLoader
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    pin_memory=False,
                    num_workers=0,
                    collate_fn=exclude_keys_collater  
                )

    @property
    def task_data(self):
        return self._data_splits

    @property
    def task_df(self):
        return self._df_splits


    @property
    def task_loader(self):
        if self._task_loader is None:
            _loader = {
                s: self._get_data_loader(
                    self.task_data[s], shuffle=(s == 'train'))
                for s in self.task_data
            }
            self._task_loader = _loader
        return self._task_loader

    def recalibrate_std(self, df, recalib_df):
        y_mean = recalib_df['y_pred'].values
        y_std = recalib_df['y_std'].values
        y_true = recalib_df['y_true'].values
        std_ratio = uct.recalibration.optimize_recalibration_ratio(
            y_mean, y_std, y_true, criterion="miscal")
        df['y_std_recalib'] = df['y_std'] * std_ratio
        return df

    def _format_predict_df(self, results,
        test_df=None, esb_yp=None, recalib_df=None):
        """
        results: dict with keys y_pred, y_true, y_var
        """
        df = self.task_df['test'].copy() if test_df is None else test_df.copy()
        assert np.allclose(results['y_true'], df['y'].values)
        df = df.rename(columns={'y': 'y_true'})
        df['y_pred'] = results['y_pred']
        if esb_yp is not None:
            if self.uncertainty:
                df['y_std'] = np.std(esb_yp, axis=0)
                if recalib_df is not None:
                    df = self.recalibrate_std(df, recalib_df)
            for i in range(self.n_ensembles):
                df[f'y_pred_{i + 1}'] = esb_yp[i]
        return df

    def train(self, n_epochs=None, patience=None,
                eval_freq=1, test_freq=None,
                monitoring_score='pearson',
                train_data=None, valid_data=None,
                rebuild_model=False,
                test_after_train=False):
        n_epochs = n_epochs or self.n_epochs
        if rebuild_model:
            self.build_model()
        tl, vl = self.task_loader['train'], self.task_loader['valid']
        rets_list = []
        for i in range(self.n_ensembles):
            stp = EarlyStopping(eval_freq=eval_freq, patience=patience,
                                    higher_better=(monitoring_score != 'mse'))
            rets = dict(
                midx = i + 1,
                model = self.models[i],
                optimizer = self.optimizers[i],
                scheduler=self.schedulers[i],
                device = self.devices[i],
                train_loader = tl,
                valid_loader = vl,
                stopper = stp,
                best_model_state_dict = None,
            )
            rets_list.append(rets)

        rets_list = Parallel(n_jobs=(self.n_ensembles if self.parallel else 1), prefer="threads")(
            delayed(_parallel_train_per_epoch)(
                kwargs=rets_list[i],
                test_loader=self.task_loader['test'],
                n_epochs=n_epochs, eval_freq=eval_freq, test_freq=test_freq,
                monitoring_score=monitoring_score,
                loss_fn=self.criterion, logger=self.logger,
                test_after_train=test_after_train,
                mode='train'
            ) for i in range(self.n_ensembles))

        # # Update models with trained versions
        for i, rets in enumerate(rets_list):
            self.models[rets['midx'] - 1] = rets['model']


    def test(self, test_model=None, test_loader=None,
                test_data=None, test_df=None,
                recalib_df=None,
                save_prediction=False, save_df_name='prediction.tsv',
                test_tag=None, print_log=False):
        test_models = self.models if test_model is None else [test_model]
        if test_data is not None:
            assert test_df is not None, 'test_df must be provided if test_data used'
            test_loader = self._get_data_loader(test_data)
        elif test_loader is not None:
            assert test_df is not None, 'test_df must be provided if test_loader used'
        else:
            test_loader = self.task_loader['test']
        rets_list = []
        for i, model in enumerate(test_models):
            rets = _parallel_test(
                kwargs={
                    'midx': i + 1,
                    'model': model,
                    'test_loader': test_loader,
                    'device': self.devices[i],
                },
                loss_fn=self.criterion,
                test_tag=f"Model {i+1}", print_log=True, logger=self.logger, mode='test'
            )
            rets_list.append(rets)


        esb_yp, esb_loss = None, 0
        for rets in rets_list:
            esb_yp = rets['y_pred'].reshape(1, -1) if esb_yp is None else\
                np.vstack((esb_yp, rets['y_pred'].reshape(1, -1)))
            esb_loss += rets['loss']

        y_true = rets['y_true']
        y_pred = np.mean(esb_yp, axis=0)
        esb_loss /= len(test_models)
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'loss': esb_loss,
        }

        eval_metrics = evaluation_metrics(
            y_true, y_pred,
            eval_metrics=['mse', 'spearman', 'pearson']
        )
        results['metrics'] = eval_metrics

        results['df'] = self._format_predict_df(results,
            test_df=test_df, esb_yp=esb_yp, recalib_df=recalib_df)
        if save_prediction:
            self.saver.save_df(results['df'], save_df_name, float_format='%g')
        if print_log:
            self.logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
        return results
