import math
import numpy as np
import torch
import torch.nn as nn
from .basic_modules import unit_tcn, unit_gcn, weights_init, bn_init, MultiScale_TemporalConv, import_class
from .hypergraph_modules import unit_hypergcn


class TCN_GCN_unit(nn.Module):
    """
    TCN-GCN融合单元
    [V5修复] 正确传递use_physical和num_joints参数到超图模块
    """

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,
                 adaptive=True, kernel_size=5, dilations=[1, 2],
                 num_hyperedges=16, use_physical=False, num_joints=None, **kwargs):
        super(TCN_GCN_unit, self).__init__()

        # 1. 标准GCN分支
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)

        # 2. 超图GCN分支(传递use_physical和num_joints参数)
        self.hypergcn1 = unit_hypergcn(
            in_channels, out_channels,
            num_hyperedges=num_hyperedges,
            use_physical=use_physical,  # [FIX] 显式传递
            num_joints=num_joints,      # [FIX] 传递关节数
            **kwargs
        )

        # 3. 自适应门控融合
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

        # 4. 多尺度时间卷积
        self.tcn1 = MultiScale_TemporalConv(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            dilations=dilations, residual=False
        )

        self.relu = nn.ReLU(inplace=False)

        # 5. 残差连接
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        z_gcn = self.gcn1(x)
        z_hyp = self.hypergcn1(x)

        # 自适应融合
        alpha = self.gate(x)
        z_fused = alpha * z_gcn + (1 - alpha) * z_hyp

        y = self.relu(self.tcn1(z_fused) + self.residual(x))
        return y


class Model(nn.Module):
    """
    DSA-HGN V5 Backbone
    [V5修复]
    1. 所有层都正确传递use_physical参数
    2. 传递num_point作为num_joints参数到超图模块
    3. 支持自动适配不同关节数的数据集
    """

    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None,
                 graph_args=dict(), in_channels=3, drop_out=0, adaptive=True,
                 num_hyperedges=16, use_physical=False,  # [FIX] 添加use_physical参数
                 base_channels=64, num_stages=10, inflate_stages=[5, 8],
                 down_stages=[5, 8], pretrained=None, data_bn_type='VC',
                 ch_ratio=2, **kwargs):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError('Graph must be specified')
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.num_class = num_class
        self.num_point = num_point

        # Data Batch Normalization
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = base_channels

        # [FIX] 共享参数字典,确保所有层配置一致
        shared_kwargs = {
            'adaptive': adaptive,
            'num_hyperedges': num_hyperedges,
            'use_physical': use_physical,
            'num_joints': num_point,  # [关键] 传递关节数
            **kwargs
        }

        # [FIX] 10层网络,全部传递use_physical和num_joints参数
        self.l1 = TCN_GCN_unit(
            in_channels, base_channel, A, residual=False, **shared_kwargs
        )

        self.l2 = TCN_GCN_unit(
            base_channel, base_channel, A, **shared_kwargs
        )

        self.l3 = TCN_GCN_unit(
            base_channel, base_channel, A, **shared_kwargs
        )

        self.l4 = TCN_GCN_unit(
            base_channel, base_channel, A, **shared_kwargs
        )

        self.l5 = TCN_GCN_unit(
            base_channel, base_channel * 2, A, stride=2, **shared_kwargs
        )

        self.l6 = TCN_GCN_unit(
            base_channel * 2, base_channel * 2, A, **shared_kwargs
        )

        self.l7 = TCN_GCN_unit(
            base_channel * 2, base_channel * 2, A, **shared_kwargs
        )

        self.l8 = TCN_GCN_unit(
            base_channel * 2, base_channel * 4, A, stride=2, **shared_kwargs
        )

        self.l9 = TCN_GCN_unit(
            base_channel * 4, base_channel * 4, A, **shared_kwargs
        )

        self.l10 = TCN_GCN_unit(
            base_channel * 4, base_channel * 4, A, **shared_kwargs
        )

        # Classification Head
        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

        bn_init(self.data_bn, 1)

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, drop=False, return_features=False):
        """
        前向传播
        Args:
            x: (N, C, T, V, M) 或 (N, T, VC)
            drop: 是否返回dropout前后的特征
            return_features: 是否返回特征向量
        Returns:
            根据参数返回不同内容
        """
        # [FIX] 仅替换NaN,不截断数值范围
        x = torch.nan_to_num(x, nan=0.0)

        # 兼容性处理:如果输入是 (N, T, VC) 格式,自动调整为 (N, C, T, V, M)
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)

        N, C, T, V, M = x.size()

        # 数据预处理与归一化 (Batch Norm)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # 通过 10 层 TCN-GCN 单元
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        z = self.l10(x)

        # 全局平均池化 (Global Average Pooling)
        c_new = z.size(1)
        x_gap = z.view(N, M, c_new, -1)
        x_gap = x_gap.mean(3).mean(1)

        # Dropout
        features_before_drop = x_gap
        x_out = self.drop_out(x_gap)

        # 根据参数返回不同内容
        if return_features:
            return x_out, z  # 返回特征向量和特征图 (用于 KD 特征对齐)

        if drop:
            return features_before_drop, x_out  # 返回 Dropout 前后的特征 (用于对比学习)
        else:
            return self.fc(x_out)  # 返回分类结果 (用于正常训练/推理)


class ChannelDifferentialBlock(nn.Module):
    """
    通道差分块(用于DualBranch模型)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.diff_conv = nn.Sequential(
            nn.Conv2d(in_channels - 1, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_diff = x[:, 1:, :, :] - x[:, :-1, :, :]
        out = self.diff_conv(x_diff)
        return out


class DualBranchDSA_HGN(nn.Module):
    """
    双分支DSA-HGN(原始+差分)
    """

    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None,
                 graph_args=dict(), in_channels=3, **kwargs):
        super().__init__()

        self.st_branch = Model(
            num_class=num_class, num_point=num_point, num_person=num_person,
            graph=graph, graph_args=graph_args, in_channels=in_channels, **kwargs
        )

        self.diff_prep = ChannelDifferentialBlock(in_channels)
        self.diff_branch = Model(
            num_class=num_class, num_point=num_point, num_person=num_person,
            graph=graph, graph_args=graph_args, in_channels=in_channels, **kwargs
        )

        base_channel = kwargs.get('base_channels', 64)
        feature_dim = base_channel * 4

        self.fusion_fc = nn.Linear(feature_dim * 2, num_class)

    def forward(self, x, drop=False, return_features=False):
        x_st = x

        N, C, T, V, M = x.shape
        x_reshaped = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x_diff = self.diff_prep(x_reshaped)
        x_diff = x_diff.view(N, M, C, T, V).permute(0, 2, 3, 4, 1).contiguous()

        feat_st, z_st = self.st_branch(x_st, return_features=True)
        feat_diff, z_diff = self.diff_branch(x_diff, return_features=True)

        feat_fused = torch.cat([feat_st, feat_diff], dim=1)

        if return_features:
            return feat_fused, z_st

        out = self.fusion_fc(feat_fused)
        return out


class HypergraphAttentionFusion(nn.Module):
    """
    超图注意力融合模块(用于多流模型)
    """

    def __init__(self, in_channels, num_streams=4):
        super().__init__()
        self.num_streams = num_streams

        self.attn_conv = nn.Sequential(
            nn.Linear(in_channels * num_streams, in_channels * num_streams // 2),
            nn.ReLU(),
            nn.Linear(in_channels * num_streams // 2, num_streams),
            nn.Softmax(dim=1)
        )

    def forward(self, features_list):
        features_stack = torch.stack(features_list, dim=1)
        features_cat = torch.cat(features_list, dim=1)

        attn_weights = self.attn_conv(features_cat)

        attn_weights = attn_weights.unsqueeze(-1)
        fused_feature = (features_stack * attn_weights).sum(dim=1)

        return fused_feature, attn_weights


class MultiStreamDSA_HGN(nn.Module):
    """
    多流DSA-HGN(Joint, Bone, Motion等)
    """

    def __init__(self, model_args, num_class=14, streams=['joint', 'bone', 'joint_motion', 'bone_motion']):
        super().__init__()
        self.streams = streams
        self.num_streams = len(streams)

        self.backbones = nn.ModuleList([
            DualBranchDSA_HGN(num_class=num_class, **model_args)
            for _ in range(self.num_streams)
        ])

        base_channel = model_args.get('base_channels', 64)
        feature_dim = base_channel * 4 * 2

        self.hafm = HypergraphAttentionFusion(feature_dim, num_streams=self.num_streams)
        self.fc = nn.Linear(feature_dim, num_class)

        self.bone_pairs = []
        if 'graph' in model_args:
            Graph = import_class(model_args['graph'])
            graph_args = model_args.get('graph_args', {})
            graph = Graph(**graph_args)
            if hasattr(graph, 'inward'):
                self.bone_pairs = graph.inward
            else:
                print("Warning: Graph does not have 'inward' attribute. Bone stream will be zero.")

    def forward(self, x_joint):
        inputs = []
        inputs.append(x_joint)

        x_bone = None
        if self.num_streams > 1:
            x_bone = torch.zeros_like(x_joint)
            if self.bone_pairs:
                for v1, v2 in self.bone_pairs:
                    x_bone[:, :, :, v1, :] = x_joint[:, :, :, v1, :] - x_joint[:, :, :, v2, :]
            inputs.append(x_bone)

        if self.num_streams > 2:
            x_jm = torch.zeros_like(x_joint)
            x_jm[:, :, :-1, :, :] = x_joint[:, :, 1:, :, :] - x_joint[:, :, :-1, :, :]
            inputs.append(x_jm)

        if self.num_streams > 3:
            if x_bone is None:
                x_bone = torch.zeros_like(x_joint)
                if self.bone_pairs:
                    for v1, v2 in self.bone_pairs:
                        x_bone[:, :, :, v1, :] = x_joint[:, :, :, v1, :] - x_joint[:, :, :, v2, :]

            x_bm = torch.zeros_like(x_bone)
            x_bm[:, :, :-1, :, :] = x_bone[:, :, 1:, :, :] - x_bone[:, :, :-1, :, :]
            inputs.append(x_bm)

        features = []
        for i, backbone in enumerate(self.backbones):
            if i < len(inputs):
                feat, _ = backbone(inputs[i], return_features=True)
                features.append(feat)

        fused_feat, attn = self.hafm(features)
        out = self.fc(fused_feat)

        return out