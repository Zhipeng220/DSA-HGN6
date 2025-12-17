import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import bn_init


class DifferentiableSparseHypergraph(nn.Module):
    """
    [Optimization] Entropy-Regularized Softmax Hypergraph Generator
    Replaces Hard Top-K with Softmax + Entropy Minimization.
    This ensures gradients flow to all prototypes while inducing sparsity via loss.
    """

    def __init__(self, in_channels, num_hyperedges, ratio=8, use_virtual_conn=True, **kwargs):
        super(DifferentiableSparseHypergraph, self).__init__()
        self.num_hyperedges = num_hyperedges
        self.in_channels = in_channels
        self.use_virtual_conn = use_virtual_conn

        # [NOTE] k_neighbors arg is kept for compatibility but not used for hard truncation anymore.

        inter_channels = max(1, in_channels // ratio)
        self.inter_channels = inter_channels

        # 1. Feature Projection
        self.query = nn.Conv2d(in_channels, inter_channels, 1)

        # 2. Prototypes (Keep Orthogonal Initialization)
        prototypes = torch.randn(inter_channels, num_hyperedges)
        if inter_channels >= num_hyperedges:
            q, _ = torch.linalg.qr(prototypes)
            self.key_prototypes = nn.Parameter(q.contiguous())
        else:
            q, _ = torch.linalg.qr(prototypes.T)
            self.key_prototypes = nn.Parameter(q.T.contiguous())

        print(f"[INIT] Softmax Entropy-Regularized Hypergraph. M={num_hyperedges}")

        # Cache for loss calculation
        self.last_h = None

    def forward(self, x):
        if not self.use_virtual_conn:
            N, C, T, V = x.shape
            return torch.zeros(N, V, self.num_hyperedges, device=x.device)

        N, C, T, V = x.shape

        # 1. Node Embedding
        q_node = self.query(x)  # (N, C', T, V)
        q_node_pooled = q_node.mean(2)  # (N, C', V)

        # L2 Normalization
        q_node_pooled = F.normalize(q_node_pooled, p=2, dim=1)  # (N, C', V)
        q_node_pooled = q_node_pooled.permute(0, 2, 1)  # (N, V, C')

        # 2. Raw Affinity
        k = self.key_prototypes  # (C', M)
        scale = self.inter_channels ** -0.5
        H_raw = torch.matmul(q_node_pooled, k) * scale  # (N, V, M)

        # =======================================================
        # [CORE CHANGE] Softmax (Full Connectivity)
        # =======================================================
        # No Top-K hard selection here. We allow gradient flow to ALL prototypes.
        # Sparsity is enforced later by Entropy Loss.
        H_final = torch.softmax(H_raw, dim=-1)

        # Save state for loss computation
        self.last_h = H_final

        return H_final  # (N, V, M)

    def get_loss(self):
        """
        Returns:
            loss_entropy: Penalizes high entropy (uniform distribution), encourages sparsity.
            loss_ortho: Enforces diversity among prototypes.
        """
        if not self.use_virtual_conn or self.last_h is None:
            dev = self.key_prototypes.device
            return torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)

        # 1. Entropy Loss (The "Soft" Top-K)
        # Minimize -sum(p * log(p)).
        # Perfect sparsity (one-hot) has entropy 0. Uniform distribution has max entropy.
        H = self.last_h
        # Add epsilon to prevent log(0)
        entropy = -torch.sum(H * torch.log(H + 1e-6), dim=-1).mean()

        # 2. Orthogonality Loss
        k = self.key_prototypes
        k_norm = F.normalize(k, p=2, dim=0)
        gram = torch.matmul(k_norm.T, k_norm)

        identity = torch.eye(gram.shape[0], device=gram.device)
        off_diagonal = gram * (1 - identity)
        loss_ortho = torch.mean(off_diagonal ** 2)

        return entropy, loss_ortho


class unit_hypergcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges=16, residual=True, **kwargs):
        super(unit_hypergcn, self).__init__()

        # Pass kwargs (like k_neighbors) down, though DifferentiableSparseHypergraph might ignore some now
        self.dhg = DifferentiableSparseHypergraph(in_channels, num_hyperedges, **kwargs)

        self.conv_v2e = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_e = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        bn_init(self.bn, 1e-5)

    def forward(self, x):
        N, C, T, V = x.shape

        # 1. Dynamic Incidence Matrix
        H = self.dhg(x)

        # 2. Hypergraph Convolution
        # Add epsilon for numerical stability
        H_norm_v = H / (H.sum(dim=1, keepdim=True) + 1e-5)

        x_v2e_feat = self.conv_v2e(x)
        x_edge = torch.einsum('nctv,nve->ncte', x_v2e_feat, H_norm_v)

        x_e_feat = self.conv_e(x_edge)

        H_norm_e = H / (H.sum(dim=2, keepdim=True) + 1e-5)
        x_node = torch.einsum('ncte,nev->nctv', x_e_feat, H_norm_e.transpose(1, 2))

        # 3. Residual & Activation
        y = self.bn(x_node)
        y = y + self.down(x)
        y = self.relu(y)
        return y