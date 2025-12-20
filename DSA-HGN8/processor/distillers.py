"""
Knowledge Distillation Modules for DSA-HGN V6
包含三种互补的蒸馏机制：
1. LogitDistiller: 基于温度缩放的Logits蒸馏
2. FeatureDistiller: 带投影器的特征对齐
3. RelationalDistiller: 样本间关系蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitDistiller(nn.Module):
    """
    Logits 知识蒸馏器
    使用温度缩放的 KL 散度损失来对齐学生和教师的预测分布

    Args:
        T (float): 温度参数，用于软化概率分布 (默认: 4.0)
    """

    def __init__(self, T=4.0):
        super(LogitDistiller, self).__init__()
        self.T = T

    def forward(self, student_logits, teacher_logits):
        """
        计算 KL 散度蒸馏损失

        Args:
            student_logits (Tensor): 学生模型输出的 logits, shape: (N, C)
            teacher_logits (Tensor): 教师模型输出的 logits, shape: (N, C)

        Returns:
            loss (Tensor): KL 散度损失 × T²
        """
        # 温度缩放后的软化概率分布
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)

        # KL散度损失 (使用 batchmean 以保持与 CrossEntropy 相同的尺度)
        loss_kd = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.T ** 2)

        return loss_kd


class FeatureDistiller(nn.Module):
    """
    特征知识蒸馏器
    通过投影器将学生特征映射到教师特征空间，使用余弦相似度对齐

    Args:
        student_dim (int): 学生特征通道数
        teacher_dim (int): 教师特征通道数
    """

    def __init__(self, student_dim, teacher_dim):
        super(FeatureDistiller, self).__init__()

        # 特征投影器: 1x1卷积 + BatchNorm + ReLU
        self.projector = nn.Sequential(
            nn.Conv2d(student_dim, teacher_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_dim),
            nn.ReLU(inplace=True)
        )

        # 初始化投影器参数
        self._init_weights()

    def _init_weights(self):
        """使用 Kaiming 初始化投影器参数"""
        for m in self.projector.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, student_feats, teacher_feats):
        """
        计算特征对齐损失

        Args:
            student_feats (Tensor): 学生特征图, shape: (N, C_s, T, V)
            teacher_feats (Tensor): 教师特征图, shape: (N, C_t, T, V)

        Returns:
            loss (Tensor): 余弦相似度损失 (1 - cosine_similarity)
        """
        # 投影学生特征到教师特征空间
        student_feats_proj = self.projector(student_feats)

        # 展平特征图为向量: (N, C, T, V) -> (N, C*T*V)
        N = student_feats_proj.size(0)
        student_flat = student_feats_proj.view(N, -1)
        teacher_flat = teacher_feats.view(N, -1)

        # 计算余弦相似度 (沿特征维度)
        cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1)

        # 损失 = 1 - cosine_similarity (越小越好)
        loss_feat = (1.0 - cos_sim).mean()

        return loss_feat


class RelationalDistiller(nn.Module):
    def __init__(self):
        super(RelationalDistiller, self).__init__()

    def forward(self, student_features, teacher_features):
        # [FIX] 如果输入是 4D (N, C, T, V), 则通过池化转换为 (N, C)
        if student_features.dim() == 4:
            student_features = F.adaptive_avg_pool2d(student_features, (1, 1)).view(student_features.size(0), -1)
        if teacher_features.dim() == 4:
            teacher_features = F.adaptive_avg_pool2d(teacher_features, (1, 1)).view(teacher_features.size(0), -1)

        # Step 1: L2 归一化
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)

        # Step 2: 计算样本间关系矩阵 (N, N)
        student_relation = torch.mm(student_norm, student_norm.t())
        teacher_relation = torch.mm(teacher_norm, teacher_norm.t())

        # Step 3: 计算 MSE 损失
        loss_rkd = F.mse_loss(student_relation, teacher_relation)
        return loss_rkd


# ============= 测试代码 =============
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Knowledge Distillation Modules")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(42)

    # 模拟数据
    batch_size = 8
    num_classes = 14
    student_channels = 64
    teacher_channels = 128
    T, V = 180, 22

    # ===== Test 1: LogitDistiller =====
    print("\n[Test 1] LogitDistiller")
    logit_distiller = LogitDistiller(T=4.0)

    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)

    loss_kd = logit_distiller(student_logits, teacher_logits)
    print(f"  Input shapes: student={student_logits.shape}, teacher={teacher_logits.shape}")
    print(f"  KD Loss: {loss_kd.item():.4f}")

    # ===== Test 2: FeatureDistiller =====
    print("\n[Test 2] FeatureDistiller")
    feature_distiller = FeatureDistiller(student_channels, teacher_channels)

    student_feats = torch.randn(batch_size, student_channels, T, V)
    teacher_feats = torch.randn(batch_size, teacher_channels, T, V)

    loss_feat = feature_distiller(student_feats, teacher_feats)
    print(f"  Input shapes: student={student_feats.shape}, teacher={teacher_feats.shape}")
    print(f"  Feature Loss: {loss_feat.item():.4f}")
    print(f"  Projector parameters: {sum(p.numel() for p in feature_distiller.parameters()):,}")

    # ===== Test 3: RelationalDistiller =====
    print("\n[Test 3] RelationalDistiller")
    rkd_distiller = RelationalDistiller()

    feat_dim = 256
    student_vec = torch.randn(batch_size, feat_dim)
    teacher_vec = torch.randn(batch_size, feat_dim)

    loss_rkd = rkd_distiller(student_vec, teacher_vec)
    print(f"  Input shapes: student={student_vec.shape}, teacher={teacher_vec.shape}")
    print(f"  RKD Loss: {loss_rkd.item():.4f}")

    # ===== Test 4: 梯度回传测试 =====
    print("\n[Test 4] Gradient Backpropagation")
    student_logits.requires_grad = True
    loss_total = (
        logit_distiller(student_logits, teacher_logits) +
        feature_distiller(student_feats, teacher_feats) +
        rkd_distiller(student_vec, teacher_vec)
    )
    loss_total.backward()
    print(f"  Total loss: {loss_total.item():.4f}")
    print(f"  Student logits grad: {student_logits.grad is not None}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)