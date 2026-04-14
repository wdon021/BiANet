# -*- coding: utf-8 -*-
"""pdb_graph_pe.py

"""

"""
Adapted from
https://github.com/jingraham/neurips19-graph-protein-design
https://github.com/drorlab/gvp-pytorch
"""
import math
import numpy as np
import scipy as sp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_geometric
import torch_cluster
from constants import LETTER_TO_NUM





AA = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']
AA2IDX = {a:i for i,a in enumerate(AA)}

ALIPHATIC = set(['A','I','L','M','V'])
AROMATIC  = set(['F','W','Y'])
POLAR     = set(['C','N','Q','S','T'])
ACIDIC    = set(['D','E'])
BASIC     = set(['H','K','R'])

def _normalize_dict(d):
    """

    """
    v = torch.tensor([d[a] for a in AA[:-1]], dtype=torch.float32)  # exclude 'X' for minmax
    mn, mx = v.min(), v.max()
    den = (mx - mn) if (mx - mn) > 0 else 1.0
    out = {a: (d[a]-mn)/den for a in AA[:-1]}
    out['X'] = float(mn + mx)/2.0  # mid for unknown
    return out



res_weight_table = _normalize_dict({'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18})

res_pka_table = _normalize_dict({'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32})

res_pkb_table = _normalize_dict({'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62})

res_pkx_table = _normalize_dict({'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00})

res_pl_table = _normalize_dict({'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96})

res_hydrophobic_ph2_table = _normalize_dict({'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49})

res_hydrophobic_ph7_table = _normalize_dict({'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63})


AROMATIC = set(['F','W','Y'])

def local_aromatic_density(X_ca: torch.Tensor,
                           seq_str: str,
                           mask: torch.Tensor,
                           cutoff: float = 8.0,
                           ref: float = 6.0) -> torch.Tensor:
    """

    """
    vmask = mask.bool()
    N = len(seq_str)
    if vmask.sum() == 0:
        return torch.zeros((N,1), dtype=torch.float32, device=X_ca.device)

    # which residues are aromatic?
    arom = torch.tensor([aa in AROMATIC for aa in seq_str],
                        dtype=torch.bool, device=X_ca.device)
    arom_valid = arom & vmask

    X = X_ca[vmask]                              # [Nv,3]
    D = torch.cdist(X, X)                        # [Nv,Nv]
    A = (D < cutoff).float()
    A.fill_diagonal_(0.0)

    # build an index map from compact Nv back to full N
    idx_map = torch.nonzero(vmask, as_tuple=False).squeeze(-1)
    arom_compact = arom_valid[idx_map].float()   # [Nv]

    # for each residue i, sum neighbors j that are aromatic
    counts = (A * arom_compact[None, :]).sum(dim=-1)  # [Nv]

    # scatter back to full length and normalize
    out = torch.zeros((N,1), dtype=torch.float32, device=X_ca.device)
    out[idx_map, 0] = (counts / ref).clamp_(max=1.0)
    return out

def residue_physchem_vector(res_char: str, device=None) -> torch.Tensor:
    """

    """
    a = res_char if res_char in AA else 'X'
    c1 = torch.tensor([
        1 if a in ALIPHATIC else 0,
        1 if a in AROMATIC  else 0,
        1 if a in POLAR     else 0,
        1 if a in ACIDIC    else 0,
        1 if a in BASIC     else 0
    ], dtype=torch.float32, device=device)
    c2 = torch.tensor([
        res_weight_table[a], res_pka_table[a], res_pkb_table[a], res_pkx_table[a], res_pl_table[a], res_hydrophobic_ph2_table[a], res_hydrophobic_ph7_table[a]
    ], dtype=torch.float32, device=device)
    return torch.cat([c1, c2], dim=0)  # [12]


AA_PKA_ACID = {'D': 3.9, 'E': 4.3, 'C': 8.3, 'Y': 10.1}  # acids: negative when deprotonated
AA_PKA_BASE = {'K':10.5, 'R':12.5, 'H': 6.0}             # bases: positive when protonated

def hh_fractional_charge(a: str, pH: float = 7.4,is_disulfide: bool = False) -> float:
    if a == 'C' and is_disulfide:
        return 0.0
    
    if a in AA_PKA_ACID:   # charge ~ -1 * fraction deprotonated
        pKa = AA_PKA_ACID[a]
        frac = 1.0 / (1.0 + 10.0**(pKa - pH))
        return -frac
    if a in AA_PKA_BASE:   # charge ~ +1 * fraction protonated
        pKa = AA_PKA_BASE[a]
        frac = 1.0 / (1.0 + 10.0**(pH - pKa))
        return  frac
    
    return 0.0

def simple_charge_at_pH7p4(a: str) -> float:
    """
  
    """
    if a == 'D': return -0.9  # pKa ~3.9, mostly deprotonated
    if a == 'E': return -0.9  # pKa ~4.3
    if a == 'K': return 0.9   # pKa ~10.5, mostly protonated
    if a == 'R': return 1.0   # pKa ~12.5
    if a == 'H': return 0.1   # pKa ~6.0, partially protonated
    if a == 'C': return -0.1  # pKa ~8.3, slightly deprotonated
    return 0.0


def contact_density(X_ca: torch.Tensor, mask: torch.Tensor, cutoff: float = 8.0, REF_MIN: float = 8) -> torch.Tensor:
    """
  
    """
    # X_ca: [N,3], mask: [N] (1 valid, 0 padded)
    vmask = mask.bool()
    X = X_ca[vmask]
    if X.shape[0] == 0:
        return torch.zeros_like(mask)
    D = torch.cdist(X, X)  # [Nv,Nv]
    A = (D < cutoff).float()
    A.fill_diagonal_(0.0)
    counts = A.sum(dim=-1)                  # [Nv]
    # backscatter into full length:
    out = torch.zeros_like(mask, dtype=torch.float32)
    # out[vmask] = counts / counts.clamp_min(1).max()  # normalize to [0,1] within pocket
    ref = torch.maximum(counts.quantile(0.95), torch.tensor(REF_MIN, device=counts.device))
    out[vmask] = (counts / ref).clamp(max=1.0)
    # Option 1: Use a fixed reference (e.g., 15 contacts is "saturated")
    # out[vmask] = counts / 15.0  # clamp to [0,1] later
    # out[vmask] = counts / counts.quantile(0.95).clamp_min(1)
    return out

HELIX_PRONE = set(['A','E','L','M'])
SHEET_PRONE = set(['V','I','Y','F'])
LOOP_PRONE  = set(['G','P','S','D'])

def ss_propensity(a: str, device=None) -> torch.Tensor:
    """
    Return a 3D vector [helix, sheet, loop] for residue 'a'.
    Overlaps resolve naturally (rare); you can also allow multiple 1's.
    # Note: these are coarse, overlapping propensities (soft priors), not ground-truth SS.
    # e.g., E is labeled helix-prone here; some residues can form multiple contexts.
    """
    return torch.tensor([
        1.0 if a in HELIX_PRONE else 0.0,
        1.0 if a in SHEET_PRONE else 0.0,
        1.0 if a in LOOP_PRONE  else 0.0
    ], dtype=torch.float32, device=device)


def pdb_to_graphs(prot_data, params):
    """

    """
    graphs = {}
    for key, struct in tqdm(prot_data.items(), desc='pdb'):
        graphs[key] = featurize_protein_graph(
            struct, name=key, **params)
    return graphs

def featurize_protein_graph(
        protein, name=None,
        num_pos_emb=16, num_rbf=16,
        contact_cutoff=8.,
    ):
    """
    Parameters: see comments of DTATask() in dta.py
    """
    with torch.no_grad():
        coords = torch.as_tensor(protein['coords'], dtype=torch.float32)
        len(coords)
        seq = torch.as_tensor([LETTER_TO_NUM[a] for a in protein['seq']], dtype=torch.long)
        seq_emb = torch.load(protein['embed']) if 'embed' in protein else None

         # Load full sequence embedding if available and remove the first dimension
        if 'full_seq_embed' in protein:
            full_seq_emb = torch.load(protein['full_seq_embed'])
            # Remove the first dimension (which is always 1)
            full_seq_emb = full_seq_emb.squeeze(0)  # Convert from [1, seq_len, 1280] to [seq_len, 1280]
        else:
            full_seq_emb = None

        mask = torch.isfinite(coords.sum(dim=(1,2)))
        coords[~mask] = np.inf

        X_ca = coords[:, 1]
        ca_mask = torch.isfinite(X_ca.sum(dim=(1)))
        ca_mask = ca_mask.float()
        ca_mask_2D = torch.unsqueeze(ca_mask, 0) * torch.unsqueeze(ca_mask, 1)
        dX_ca = torch.unsqueeze(X_ca, 0) - torch.unsqueeze(X_ca, 1)
        D_ca = ca_mask_2D * torch.sqrt(torch.sum(dX_ca**2, 2) + 1e-6)
        edge_index = torch.nonzero((D_ca < contact_cutoff) & (ca_mask_2D == 1))
        edge_index = edge_index.t().contiguous()


        O_feature = _local_frame(X_ca, edge_index)
        pos_embeddings = _positional_embeddings(edge_index, num_embeddings=num_pos_emb)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf)

        dihedrals = _dihedrals(coords)
        orientations = _orientations(X_ca)
        sidechains = _sidechains(coords)

        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, O_feature, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)
        # print(f"node_s shape: {node_s.shape}, node_v shape: {node_v.shape}, edge_s shape: {edge_s.shape}, edge_v shape: {edge_v.shape}")
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                (node_s, node_v, edge_s, edge_v))

        seq_str = protein['seq']                  # string of length N
        N = len(seq_str)
        device = X_ca.device
        # 1) 12-d sequence physchem per residue
        physchem = torch.stack([residue_physchem_vector(a) for a in seq_str], dim=0)  # [N, 12]
        
        # 2) simple charge at pH 7.4 (1 dim)
        # charge = torch.tensor([simple_charge_at_pH7p4(a) for a in seq_str], dtype=torch.float32).unsqueeze(-1)  # [N,1]
        fractional_charge = torch.tensor([hh_fractional_charge(a, pH=7.4) for a in seq_str], dtype=torch.float32).unsqueeze(-1)# [N,1]

        # 3) structure-based exposure proxy (contact density) (1 dim)
        exposure = contact_density(X_ca, mask.float())[:, None]  # [N,1]
        arom_density = local_aromatic_density(X_ca, protein['seq'], mask, cutoff=8.0, ref=6.0)  # [N,1]
        ss_prop = torch.stack([ss_propensity(a, device=device) for a in seq_str], dim=0)  # [N,3]

        # mask invalid rows to zero
        physchem[~mask] = 0.0
        # charge[~mask]   = 0.0
        exposure[~mask] = 0.0
        arom_density[~mask] = 0.0
        ss_prop[~mask] = 0.0
        fractional_charge[~mask] = 0.0
        
        # concatenate to node_s
        pocket_scalar_feats = torch.cat([physchem, exposure, arom_density, ss_prop,fractional_charge], dim=-1)  # [N, 12+1+1+3+1=18]
        node_s = torch.cat([node_s, pocket_scalar_feats], dim=-1) #[N, 6+14] = [N,20]


    data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,
                                        node_s=node_s, node_v=node_v,
                                        edge_s=edge_s, edge_v=edge_v,
                                        edge_index=edge_index, mask=mask,
                                        seq_emb=seq_emb,
                                        full_seq_emb=full_seq_emb)
    if data.x.shape[0] != data.seq.shape[0]:
        print(f"Length mismatch detected for protein: {name}")
        print(f"x length: {data.x.shape[0]}, seq length: {data.seq.shape[0]}")
        print(f"Protein sequence: {protein['seq']}")
        print(f"Mask: {mask}")
        if seq_emb is not None:
            print(f"seq_emb length: {data.seq_emb.shape[0]}")
        if full_seq_emb is not None:
            print(f"full_seq_emb shape: {data.full_seq_emb.shape}")  # Print the shape after squeezing
    return data


def _dihedrals(X, eps=1e-7):
    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _positional_embeddings(edge_index,
                            num_embeddings=None,
                            period_range=[2, 1000]):
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def _orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def _local_frame(X, edge_index, eps=1e-6):
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    o_1 = _normalize(u_2 - u_1, dim=-1)
    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 1)
    O = F.pad(O, (0, 0, 0, 0, 1, 2), 'constant', 0)

    # dX = X[edge_index[0]] - X[edge_index[1]]
    dX = X[edge_index[1]] - X[edge_index[0]]
    dX = _normalize(dX, dim=-1)
    # dU = torch.bmm(O[edge_index[1]], dX.unsqueeze(2)).squeeze(2)
    dU = torch.bmm(O[edge_index[0]], dX.unsqueeze(2)).squeeze(2)
    R = torch.bmm(O[edge_index[0]].transpose(-1,-2), O[edge_index[1]])
    Q = _quaternions(R)
    O_features = torch.cat((dU,Q), dim=-1)

    return O_features


def _quaternions(R):
    # Simple Wikipedia version
    # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:, i, j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q
