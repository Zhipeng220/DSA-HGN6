import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .basic_modules import bn_init


class DifferentiableSparseHypergraph(nn.Module):
    """
    Entropy-Regularized Softmax Hypergraph Generator
    使用Softmax + 熵正则化实现可微分的稀疏超图
    """

    def __init__(self, in_channels, num_hyperedges, ratio=8, use_virtual_conn=True, **kwargs):
        super(DifferentiableSparseHypergraph, self).__init__()
        self.num_hyperedges = num_hyperedges
        self.in_channels = in_channels
        self.use_virtual_conn = use_virtual_conn

        inter_channels = max(1, in_channels // ratio)
        self.inter_channels = inter_channels

        # 1. Feature Projection
        self.query = nn.Conv2d(in_channels, inter_channels, 1)

        # 2. Learnable Prototypes (正交初始化)
        prototypes = torch.randn(inter_channels, num_hyperedges)
        if inter_channels >= num_hyperedges:
            q, _ = torch.linalg.qr(prototypes)
            self.key_prototypes = nn.Parameter(q.contiguous())
        else:
            q, _ = torch.linalg.qr(prototypes.T)
            self.key_prototypes = nn.Parameter(q.T.contiguous())

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

        # 2. Compute Affinity
        k = self.key_prototypes  # (C', M)
        scale = self.inter_channels ** -0.5
        H_raw = torch.matmul(q_node_pooled, k) * scale  # (N, V, M)

        # 3. Softmax (允许梯度流向所有原型)
        H_final = torch.softmax(H_raw, dim=-1)

        # Save for loss computation
        self.last_h = H_final

        return H_final  # (N, V, M)

    def get_loss(self):
        """
        Returns:
            loss_entropy: 惩罚高熵(均匀分布),鼓励稀疏性
            loss_ortho: 强制原型之间的多样性
        """
        if not self.use_virtual_conn or self.last_h is None:
            dev = self.key_prototypes.device
            return torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)

        # 1. Entropy Loss (软Top-K)
        H = self.last_h
        entropy = -torch.sum(H * torch.log(H + 1e-6), dim=-1).mean()

        # 2. Orthogonality Loss
        k = self.key_prototypes
        k_norm = F.normalize(k, p=2, dim=0)
        gram = torch.matmul(k_norm.T, k_norm)

        identity = torch.eye(gram.shape[0], device=gram.device)
        off_diagonal = gram * (1 - identity)
        loss_ortho = torch.mean(off_diagonal ** 2)

        return entropy, loss_ortho


class PhysicallyInformedHypergraph(nn.Module):
    """
    [V5核心创新 - 已修复] 物理感知超图
    融合静态物理先验(手指拓扑)和动态学习的超图结构

    修复内容:
    1. alpha从标量改为向量,每个超边独立学习融合权重
    2. 支持自动检测关节数,不再硬编码为22
    3. 未使用的静态边初始化为0而非随机噪声
    """

    def __init__(self, in_channels, num_hyperedges, num_joints=None,
                 num_static_edges=5, init_alpha=0.5, ratio=8,
                 use_virtual_conn=True, **kwargs):
        """
        Args:
            in_channels: 输入通道数
            num_hyperedges: 总超边数(必须 >= num_static_edges)
            num_joints: 关节数(None则自动检测,推荐显式传入)
            num_static_edges: 物理先验边数(5=手指, 7=手指+手腕+手掌)
            init_alpha: 动态分支的初始权重(0.5表示均等融合)
            ratio: 特征压缩比例
            use_virtual_conn: 是否使用虚拟连接
        """
        super(PhysicallyInformedHypergraph, self).__init__()

        if num_hyperedges < num_static_edges:
            raise ValueError(f"num_hyperedges ({num_hyperedges}) must be >= num_static_edges ({num_static_edges})")

        self.num_hyperedges = num_hyperedges
        self.num_static_edges = num_static_edges
        self.use_virtual_conn = use_virtual_conn
        self.num_joints = num_joints  # 允许None,首次forward时自动检测

        # [关键修复] 使用flag标记而不是设置H_static=None
        self._h_static_initialized = False
        self.static_topology = self._get_static_topology(num_static_edges)

        # 动态分支
        self.dynamic_branch = DifferentiableSparseHypergraph(
            in_channels, num_hyperedges, ratio=ratio,
            use_virtual_conn=use_virtual_conn, **kwargs
        )

        # [FIX 2] 融合参数改为向量,每个超边独立学习
        # shape: (1, 1, num_hyperedges) 以便与 (N, V, M) 广播
        init_logit = math.log(init_alpha / (1 - init_alpha))
        self.alpha_logit = nn.Parameter(
            torch.full((1, 1, num_hyperedges), init_logit)
        )

        print(f"[PhysicalHypergraph] Static edges: {num_static_edges}, "
              f"Total edges: {num_hyperedges}, Per-edge fusion: Enabled")

    def _get_static_topology(self, num_static_edges):
        """
        定义静态拓扑结构(与关节数无关的逻辑定义)
        返回: List[List[int]] - 每个超边包含的关节索引(相对索引)
        """
        if num_static_edges == 5:
            # 5个手指(不包括手腕和手掌)
            # 假设关节索引: 0=腕, 1=掌心, 2-5=拇指, 6-9=食指, ...
            return [
                [2, 3, 4, 5],  # Thumb
                [6, 7, 8, 9],  # Index
                [10, 11, 12, 13],  # Middle
                [14, 15, 16, 17],  # Ring
                [18, 19, 20, 21]  # Pinky
            ]
        elif num_static_edges == 7:
            # 包括手腕和手掌
            return [
                [0],  # Wrist (独立节点)
                [1],  # Palm (中心节点)
                [2, 3, 4, 5],  # Thumb
                [6, 7, 8, 9],  # Index
                [10, 11, 12, 13],  # Middle
                [14, 15, 16, 17],  # Ring
                [18, 19, 20, 21]  # Pinky
            ]
        else:
            raise ValueError(f"num_static_edges must be 5 or 7, got {num_static_edges}")

    def _initialize_static_hypergraph(self, num_joints, device):
        """
        [FIX 3 & 4] 自动适应关节数并初始化静态超图
        """
        H_static = torch.zeros(num_joints, self.num_hyperedges, device=device)

        # 填充静态边(使用相对索引,自动检查边界)
        for i, nodes in enumerate(self.static_topology):
            for node in nodes:
                if node < num_joints:  # 防止越界
                    H_static[node, i] = 1.0
                else:
                    print(f"[Warning] Node {node} exceeds num_joints {num_joints}, skipped")

        # [FIX 4] 剩余边初始化为0(而非随机噪声)
        # 让动态分支完全接管这些通道
        if self.num_hyperedges > self.num_static_edges:
            H_static[:, self.num_static_edges:] = 0.0

        # 归一化(避免除以零)
        H_static = H_static / (H_static.sum(dim=0, keepdim=True) + 1e-5)

        self.register_buffer('H_static', H_static)
        print(f"[PhysicalHypergraph] Static hypergraph initialized for {num_joints} joints")

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V) 输入特征
        Returns:
            H_final: (N, V, M) 融合后的超边矩阵
        """
        if not self.use_virtual_conn:
            N, C, T, V = x.shape
            return torch.zeros(N, V, self.num_hyperedges, device=x.device)

        N, C, T, V = x.shape

        # [关键修复] 使用flag检查是否已初始化
        if not self._h_static_initialized:
            if self.num_joints is None:
                self.num_joints = V
                print(f"[PhysicalHypergraph] Auto-detected num_joints: {V}")
            elif self.num_joints != V:
                print(f"[Warning] Expected {self.num_joints} joints but got {V}")

            self._initialize_static_hypergraph(V, x.device)
            self._h_static_initialized = True

        # 1. 动态超图
        H_dyn = self.dynamic_branch(x)  # (N, V, M)

        # 2. 静态物理先验
        H_phy = self.H_static.unsqueeze(0).expand(N, -1, -1)  # (N, V, M)

        # 3. 加权融合 [FIX 2] 每个超边独立权重
        # alpha_logit: (1, 1, M) -> weight: (1, 1, M) 广播到 (N, V, M)
        weight = torch.sigmoid(self.alpha_logit)
        H_final = weight * H_dyn + (1 - weight) * H_phy

        # 保存动态分支的last_h用于loss计算
        self.last_h = self.dynamic_branch.last_h

        return H_final

    def get_loss(self):
        """
        返回正则化损失
        """
        return self.dynamic_branch.get_loss()


class unit_hypergcn(nn.Module):
    """
    超图卷积单元
    支持物理感知超图(use_physical=True)或纯学习超图(use_physical=False)
    """

    def __init__(self, in_channels, out_channels, num_hyperedges=16,
                 residual=True, use_physical=False, num_joints=None, **kwargs):
        super(unit_hypergcn, self).__init__()

        # [V5关键改动] 根据use_physical选择超图类型
        if use_physical:
            self.dhg = PhysicallyInformedHypergraph(
                in_channels, num_hyperedges, num_joints=num_joints, **kwargs
            )
        else:
            self.dhg = DifferentiableSparseHypergraph(
                in_channels, num_hyperedges, **kwargs
            )

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

        # 1. 生成超边矩阵(动态或物理感知)
        H = self.dhg(x)  # (N, V, M)

        # 2. 超图卷积
        H_norm_v = H / (H.sum(dim=1, keepdim=True) + 1e-5)
        x_v2e_feat = self.conv_v2e(x)
        x_edge = torch.einsum('nctv,nve->ncte', x_v2e_feat, H_norm_v)

        x_e_feat = self.conv_e(x_edge)

        H_norm_e = H / (H.sum(dim=2, keepdim=True) + 1e-5)
        x_node = torch.einsum('ncte,nev->nctv', x_e_feat, H_norm_e.transpose(1, 2))

        # 3. 残差连接和激活
        y = self.bn(x_node)
        y = y + self.down(x)
        y = self.relu(y)
        return y

    def get_loss(self):
        """
        获取超图正则化损失
        """
        return self.dhg.get_loss()