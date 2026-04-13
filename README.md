# BiANet: Sigmoid-Gated Bidirectional Hierarchical Cross-Attention for Drug-Target Affinity Prediction

This repository contains the model code and training scripts needed to reproduce the Davis results reported in our IJCNN/WCCI 2026 paper. An extended version with additional benchmarks and analyses is in preparation.

## Installation

```bash
conda create -n bianet python=3.10
conda activate bianet
pip install -r requirements.txt
```

PyTorch Geometric dependencies (`torch-scatter`, `torch-cluster`) may require matching your CUDA version. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if `pip install` fails for these packages.

## Data

Davis preprocessing follows [KDBNet](https://github.com/luoyunan/KDBNet). You need:
- `davis_data.tsv` — affinity labels
- `davis_protein2pdb.yaml` — protein-to-PDB mapping
- `pockets_structure.json` — KLIFS pocket coordinates
- `davis_mol3d_sdf/` — PubChem 3D conformers
- `davis_cluster_id50_cluster.tsv` — MMseqs2 clustering output
- ESM-2 embeddings per pocket chain (see [ESM](https://github.com/facebookresearch/esm) for extraction)

Place these under `data/davis/` and update paths in `Data_split_Davis.py`.

## Reproducing Table I

Generate data splits (S1 random, S2 novel drug, S3 novel protein):

```bash
python Data_split_Davis.py
```

Train BiANet with the hyperparameters reported in the paper:

```bash
python run.py --base_path data/davis --seed 55
```

Results (MSE, Pearson, Spearman) are written to the output directory. Mean ± std across 5 seeds `{11, 17, 23, 29, 31}` reproduces Table I.

## Files

- `model.py` — BiANet architecture (GVP drug encoder, TransformerConv protein encoder, bidirectional sigmoid cross-attention, EBCA graph gating)
- `attention.py` — `SigmoidCrossAttention` and `EnhancedBidirectionalCrossAttentionV2` modules with stabilization (RMSNorm, length bias, LayerScale)
- `gvp.py` — Geometric Vector Perceptron layers (from Jing et al., 2021)
- `pdb_graph_pe.py`, `mol_graph_pe.py` — featurization
- `train.py`, `run.py` — training loop and entry point
- `utils.py`, `metrics.py`, `parsing.py`, `constants.py` — utilities

## Citation

```bibtex
@inproceedings{wu2026bianet,
  title={Sigmoid-Gated Bidirectional Hierarchical Cross-Attention for Drug-Target Affinity Prediction},
  author={Wu, Dongliang and Nguyen, Binh},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2026}
}
```

## License

MIT