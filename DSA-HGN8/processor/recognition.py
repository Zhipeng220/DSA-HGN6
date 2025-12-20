import sys
import argparse
import yaml
import math
import numpy as np
import os
import random

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .distillers import LogitDistiller, FeatureDistiller, RelationalDistiller  # 导入蒸馏器


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


def init_seed(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


class FT_Processor(Processor):
    """
    V6 Processor: 模块化知识蒸馏系统
    支持三种互补的蒸馏机制：
    1. Logits KD (默认开启)
    2. Feature Projection KD (可选)
    3. Relational KD (可选)
    """

    def load_model(self):
        """加载学生模型、教师模型和蒸馏器"""

        # ========== 1. 加载学生模型 ==========
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        #self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

        # ========== 2. 加载教师模型和蒸馏器 ==========
        if self.arg.phase == 'train' and hasattr(self.arg, 'teacher_weights') and self.arg.teacher_weights:
            self.io.print_log(f'\n{"=" * 60}')
            self.io.print_log(f'[KD System] Initializing Knowledge Distillation')
            self.io.print_log(f'{"=" * 60}')
            self.io.print_log(f'[KD] Teacher weights: {self.arg.teacher_weights}')

            # 2.1 构建教师模型 (Joint流配置)
            teacher_args = self.arg.model_args.copy()
            teacher_args['in_channels'] = 3  # Joint流输入
            teacher_args['use_physical'] = False  # 教师不使用物理先验

            self.teacher = self.io.load_model(self.arg.model, **teacher_args)

            # 2.2 加载教师预训练权重
            weights = torch.load(self.arg.teacher_weights, map_location='cpu')
            if isinstance(weights, dict):
                if 'model' in weights:
                    weights = weights['model']
                elif 'state_dict' in weights:
                    weights = weights['state_dict']

            # 处理 DataParallel 前缀
            from collections import OrderedDict
            new_weights = OrderedDict()
            for k, v in weights.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_weights[name] = v

            self.teacher.load_state_dict(new_weights, strict=False)

            # 2.3 移动教师到设备并设为评估模式
            if self.arg.use_gpu:
                self.teacher = self.teacher.to(self.dev)

            self.teacher.eval()

            # 修复 BatchNorm 的 running buffers
            for module in self.teacher.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if module.running_mean is not None:
                        module.running_mean = module.running_mean.to(self.dev)
                    if module.running_var is not None:
                        module.running_var = module.running_var.to(self.dev)
                    if module.num_batches_tracked is not None:
                        module.num_batches_tracked = module.num_batches_tracked.to(self.dev)

            # 冻结教师参数
            for param in self.teacher.parameters():
                param.requires_grad = False

            self.io.print_log('[KD] ✓ Teacher model loaded and frozen.')

            # ========== 3. 初始化蒸馏器 ==========
            self.io.print_log(f'\n[Distillers] Initializing distillation modules:')

            # 3.1 Logits Distiller (默认开启)
            self.logit_distiller = LogitDistiller(T=4.0)
            self.io.print_log(f'  ✓ LogitDistiller (T=4.0)')

            # 3.2 Feature Distiller (可选)
            if getattr(self.arg, 'use_feature_projector', False):
                student_dim = self.arg.model_args.get('base_channels', 64)
                teacher_dim = self.arg.model_args.get('base_channels', 64)
                self.feature_distiller = FeatureDistiller(student_dim, teacher_dim)

                if self.arg.use_gpu:
                    self.feature_distiller = self.feature_distiller.to(self.dev)

                self.io.print_log(f'  ✓ FeatureDistiller (student_dim={student_dim}, teacher_dim={teacher_dim})')
            else:
                self.feature_distiller = None
                self.io.print_log(f'  ✗ FeatureDistiller (disabled)')

            # 3.3 Relational Distiller (可选)
            if getattr(self.arg, 'use_rkd', False):
                self.rkd_distiller = RelationalDistiller()
                self.io.print_log(f'  ✓ RelationalDistiller (RKD)')
            else:
                self.rkd_distiller = None
                self.io.print_log(f'  ✗ RelationalDistiller (disabled)')

            # 打印蒸馏配置摘要
            self.io.print_log(f'\n[KD Summary] Enabled mechanisms:')
            self.io.print_log(f'  • Logits KD: YES (λ={getattr(self.arg, "lambda_kd", 0.3)})')
            self.io.print_log(f'  • Feature KD: {"YES" if self.feature_distiller else "NO"} '
                              f'(λ={getattr(self.arg, "lambda_feat", 0.1) if self.feature_distiller else 0.0})')
            self.io.print_log(f'  • Relational KD: {"YES" if self.rkd_distiller else "NO"} '
                              f'(λ={getattr(self.arg, "lambda_rkd", 1.0) if self.rkd_distiller else 0.0})')
            self.io.print_log(f'{"=" * 60}\n')
        else:
            self.teacher = None
            self.logit_distiller = None
            self.feature_distiller = None
            self.rkd_distiller = None

    def load_optimizer(self):
        """加载优化器，包含投影器参数"""

        # 收集需要优化的参数
        params_to_optimize = list(self.model.parameters())

        # 如果使用特征投影器，将其参数加入优化器
        if hasattr(self, 'feature_distiller') and self.feature_distiller is not None:
            params_to_optimize += list(self.feature_distiller.parameters())
            self.io.print_log('[Optimizer] Added FeatureDistiller parameters.')

        # 创建优化器
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params_to_optimize,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params_to_optimize,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                params_to_optimize,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
                eps=1e-4
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")

    def load_data(self):
        """加载数据并正确配置 Teacher 和 Student 的预处理"""
        self.data_loader = dict()
        train_mean_map = None
        train_std_map = None

        # 1. 加载训练集 (Bone流)
        if self.arg.train_feeder_args:
            g = torch.Generator()
            g.manual_seed(0)

            train_feeder = import_class(self.arg.train_feeder)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed,
                generator=g,
                pin_memory=True
            )

            # 获取训练集统计量
            if hasattr(self.data_loader['train'].dataset, 'mean_map'):
                train_mean_map = self.data_loader['train'].dataset.mean_map
                train_std_map = self.data_loader['train'].dataset.std_map
                self.io.print_log('[DATA] Train stats loaded from Bone stream.')

            # 2. 加载 Joint Loader (用于KD)
            if self.arg.phase == 'train' and hasattr(self.arg, 'teacher_weights') and self.arg.teacher_weights:
                joint_args = self.arg.train_feeder_args.copy()

                # 恢复 Joint 流的正确配置
                joint_args['bone'] = False
                joint_args['vel'] = False
                joint_args['normalization'] = True  # Teacher 在归一化数据上训练
                joint_args['random_rot'] = False  # Teacher 训练时未使用旋转增强

                g_joint = torch.Generator()
                g_joint.manual_seed(0)

                self.data_loader['train_joint'] = torch.utils.data.DataLoader(
                    dataset=train_feeder(**joint_args),
                    batch_size=self.arg.batch_size,
                    shuffle=True,
                    num_workers=self.arg.num_worker,
                    drop_last=True,
                    worker_init_fn=init_seed,
                    generator=g_joint,
                    pin_memory=True
                )
                self.io.print_log('[KD] Joint stream loader created.')

        # 3. 加载测试集并注入训练集统计量
        if self.arg.test_feeder_args:
            test_feeder = import_class(self.arg.test_feeder)
            test_args = self.arg.test_feeder_args.copy()

            # 注入训练集统计量到测试集
            if train_mean_map is not None:
                test_args['mean_map'] = train_mean_map
                test_args['std_map'] = train_std_map
                self.io.print_log('[DATA] Injected train stats into test feeder.')

            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=test_feeder(**test_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed,
                pin_memory=True
            )

    def train(self, epoch):
        """训练循环 - 支持多种蒸馏损失"""
        self.model.train()
        if hasattr(self, 'teacher') and self.teacher:
            self.teacher.eval()

        # 如果有特征投影器，设为训练模式
        if hasattr(self, 'feature_distiller') and self.feature_distiller:
            self.feature_distiller.train()

        self.adjust_lr()

        loader_bone = self.data_loader['train']
        loader_joint = self.data_loader.get('train_joint', None)

        if loader_joint and hasattr(self, 'teacher') and self.teacher:
            iterator = zip(loader_bone, loader_joint)
        else:
            iterator = ((batch, None) for batch in loader_bone)

        loss_value = []
        first_batch = True

        for (data_bone, label), batch_joint in iterator:
            self.global_step += 1
            data_bone = data_bone.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # ========== 学生前向传播 (支持多种返回格式) ==========
            try:
                student_output = self.model(data_bone, return_features=True)
                if isinstance(student_output, tuple):
                    if len(student_output) == 3:
                        student_logits, student_feat_vec, student_feat_map = student_output
                    elif len(student_output) == 2:
                        # [FIX] 如果只返回两个值，通常是 (logits, feature_vector)
                        student_logits, student_feat_vec = student_output
                        student_feat_map = None  # 特征图设为空
                    else:
                        student_logits = student_output[0]
                        student_feat_vec = student_feat_map = None
                else:
                    student_logits = student_output
                    student_feat_vec = student_feat_map = None
            except TypeError:
                # 模型不支持 return_features 参数
                student_logits = self.model(data_bone)
                student_feat_vec = student_feat_map = None
            # 计算交叉熵损失
            loss_ce = self.loss(student_logits, label)

            # ========== 知识蒸馏损失 ==========
            loss_kd = torch.tensor(0.0).to(self.dev)
            loss_feat = torch.tensor(0.0).to(self.dev)
            loss_rkd = torch.tensor(0.0).to(self.dev)

            if batch_joint is not None and hasattr(self, 'teacher') and self.teacher:
                data_joint, _ = batch_joint
                data_joint = data_joint.float().to(self.dev, non_blocking=True)

                # Debug 数据范围（仅第一个batch）
                if first_batch:
                    self.io.print_log(f'[DEBUG] Bone data range: [{data_bone.min():.2f}, {data_bone.max():.2f}]')
                    self.io.print_log(f'[DEBUG] Joint data range: [{data_joint.min():.2f}, {data_joint.max():.2f}]')
                    first_batch = False
                # 教师前向传播
                with torch.no_grad():
                    try:
                        teacher_output = self.teacher(data_joint, return_features=True)
                        if isinstance(teacher_output, tuple):
                            if len(teacher_output) == 3:
                                teacher_logits, teacher_feat_vec, teacher_feat_map = teacher_output
                            elif len(teacher_output) == 2:
                                teacher_logits, teacher_feat_vec = teacher_output
                                teacher_feat_map = None
                            else:
                                teacher_logits = teacher_output[0]
                                teacher_feat_vec = teacher_feat_map = None
                        else:
                            teacher_logits = teacher_output
                            teacher_feat_vec = teacher_feat_map = None
                    except TypeError:
                        teacher_logits = self.teacher(data_joint)
                        teacher_feat_vec = teacher_feat_map = None

                # 1. Logits KD (默认开启)
                if self.logit_distiller is not None:
                    loss_kd = self.logit_distiller(student_logits, teacher_logits)

                # 2. Feature KD (可选)
                if self.feature_distiller is not None and student_feat_map is not None and teacher_feat_map is not None:
                    loss_feat = self.feature_distiller(student_feat_map, teacher_feat_map)

                # 3. Relational KD (可选)
                if self.rkd_distiller is not None and student_feat_vec is not None and teacher_feat_vec is not None:
                    loss_rkd = self.rkd_distiller(student_feat_vec, teacher_feat_vec)

            # ========== 正则化损失 (Entropy + Orthogonality) ==========
            loss_entropy = torch.tensor(0.0).to(self.dev)
            loss_ortho = torch.tensor(0.0).to(self.dev)

            model_core = self.model.module if hasattr(self.model, 'module') else self.model
            for m in model_core.modules():
                if hasattr(m, 'get_loss') and callable(m.get_loss):
                    l_e, l_o = m.get_loss()
                    loss_entropy += l_e
                    loss_ortho += l_o

            # ========== 总损失 ==========
            lambda_kd = getattr(self.arg, 'lambda_kd', 0.0)
            lambda_feat = getattr(self.arg, 'lambda_feat', 0.0)
            lambda_rkd = getattr(self.arg, 'lambda_rkd', 0.0)
            lambda_ent = getattr(self.arg, 'lambda_entropy', 0.0)
            lambda_orth = getattr(self.arg, 'lambda_ortho', 0.0)

            loss = (loss_ce +
                    lambda_kd * loss_kd +
                    lambda_feat * loss_feat +
                    lambda_rkd * loss_rkd +
                    lambda_ent * loss_entropy +
                    lambda_orth * loss_ortho)

            # ========== 反向传播 ==========
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if hasattr(self.arg, 'grad_clip_norm') and self.arg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip_norm)
                if self.feature_distiller is not None:
                    torch.nn.utils.clip_grad_norm_(self.feature_distiller.parameters(), self.arg.grad_clip_norm)

            self.optimizer.step()

            # ========== 记录日志 ==========
            self.iter_info['loss'] = loss.item()
            self.iter_info['loss_ce'] = loss_ce.item()
            self.iter_info['loss_kd'] = loss_kd.item()
            self.iter_info['loss_feat'] = loss_feat.item()
            self.iter_info['loss_rkd'] = loss_rkd.item()
            self.iter_info['loss_ent'] = loss_entropy.item()
            self.iter_info['loss_orth'] = loss_ortho.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)

            loss_value.append(loss.item())

            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()

    def test(self, epoch):
        """测试循环"""
        self.model.eval()

        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            with torch.no_grad():
                # 测试时只需要 logits
                try:
                    output = self.model(data, return_features=False)
                    if isinstance(output, tuple):
                        output = output[0]  # 取 logits
                except TypeError:
                    output = self.model(data)

            result_frag.append(output.data.cpu().numpy())

            if self.arg.phase in ['train', 'test']:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if self.arg.phase in ['train', 'test']:
            self.label = np.concatenate(label_frag)

        self.eval_info['eval_mean_loss'] = np.mean(loss_value)
        self.show_eval_info()

        # Show Top-K Accuracy
        for k in self.arg.show_topk:
            self.show_topk(k)

        self.show_best(1)
        self.eval_log_writer(epoch)

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.current_result = round(accuracy, 5)

        if self.best_result <= self.current_result:
            self.best_result = self.current_result

        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def adjust_lr(self):
        lr_decay_rate = getattr(self.arg, 'lr_decay_rate', 0.1)

        # Warm up
        if hasattr(self.arg, 'warm_up_epoch') and self.arg.warm_up_epoch > 0:
            if self.meta_info['epoch'] < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (self.meta_info['epoch'] + 1) / self.arg.warm_up_epoch
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
                return

        # Step decay
        if self.arg.step:
            lr = self.arg.base_lr * (
                    lr_decay_rate ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step))
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='DSA-HGN V6 Recognition with Modular KD'
        )

        # Learning Rate
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
        parser.add_argument('--warm_up_epoch', type=int, default=0, help='warm up epochs')

        # Optimizer
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')

        # ========== Knowledge Distillation ==========
        parser.add_argument('--teacher_weights', type=str, default=None,
                            help='path to teacher model weights')

        # 蒸馏机制开关
        parser.add_argument('--use_feature_projector', type=str2bool, default=False,
                            help='enable feature projection distillation')
        parser.add_argument('--use_rkd', type=str2bool, default=False,
                            help='enable relational knowledge distillation')

        # 蒸馏损失权重
        parser.add_argument('--lambda_kd', type=float, default=0.3,
                            help='weight for logits KD loss')
        parser.add_argument('--lambda_feat', type=float, default=0.1,
                            help='weight for feature alignment loss')
        parser.add_argument('--lambda_rkd', type=float, default=1.0,
                            help='weight for relational KD loss')

        # Regularization
        parser.add_argument('--lambda_entropy', type=float, default=0.001,
                            help='weight for entropy loss')
        parser.add_argument('--lambda_ortho', type=float, default=0.1,
                            help='weight for orthogonality loss')
        parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                            help='gradient clipping norm')

        # Evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        parser.add_argument('--stream', type=str, default='joint',
                            help='the stream of input')

        return parser