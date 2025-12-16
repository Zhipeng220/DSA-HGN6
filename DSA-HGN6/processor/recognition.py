import sys
import argparse
import yaml
import math
import numpy as np
import os
import random  # [FIX] 新增导入

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # [FIX] 新增导入

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


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
    V5 Processor: Physical-Informed Hypergraph + Cross-Stream Knowledge Distillation
    """

    def load_model(self):
        # 1. Load Student Model
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

        # 2. Load Teacher Model for KD
        if self.arg.phase == 'train' and hasattr(self.arg, 'teacher_weights') and self.arg.teacher_weights:
            self.io.print_log(f'[KD] Loading Teacher from: {self.arg.teacher_weights}')

            # Teacher使用Joint流配置（3通道输入，不使用物理先验）
            teacher_args = self.arg.model_args.copy()
            teacher_args['in_channels'] = 3
            teacher_args['use_physical'] = False

            self.teacher = self.io.load_model(self.arg.model, **teacher_args)

            # 加载预训练权重
            weights = torch.load(self.arg.teacher_weights, map_location='cpu')
            if isinstance(weights, dict):
                if 'model' in weights:
                    weights = weights['model']
                elif 'state_dict' in weights:
                    weights = weights['state_dict']

            # 处理DataParallel前缀
            from collections import OrderedDict
            new_weights = OrderedDict()
            for k, v in weights.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_weights[name] = v

            self.teacher.load_state_dict(new_weights, strict=False)
            self.teacher.eval()

            # 冻结Teacher参数
            for param in self.teacher.parameters():
                param.requires_grad = False

            self.io.print_log('[KD] Teacher loaded successfully.')
        else:
            self.teacher = None

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
                eps=1e-4
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")

    def load_data(self):
        """
        [FIX] 加载数据并正确注入统计量
        """
        self.data_loader = dict()
        train_mean_map = None
        train_std_map = None

        # 1. 加载训练集（Bone流）
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

            # 2. 加载Joint Loader（用于KD）
            if self.arg.phase == 'train' and hasattr(self.arg, 'teacher_weights') and self.arg.teacher_weights:
                joint_args = self.arg.train_feeder_args.copy()
                joint_args['bone.yaml'] = False
                joint_args['vel'] = False

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

            # [CRITICAL] 注入训练集统计量到测试集
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
        """
        [FIX] 训练循环，支持KD
        """
        self.model.train()
        if hasattr(self, 'teacher') and self.teacher:
            self.teacher.eval()

        self.adjust_lr()

        loader_bone = self.data_loader['train']
        loader_joint = self.data_loader.get('train_joint', None)

        # 构建迭代器（确保样本对齐）
        if loader_joint and hasattr(self, 'teacher') and self.teacher:
            iterator = zip(loader_bone, loader_joint)
        else:
            iterator = ((batch, None) for batch in loader_bone)

        loss_value = []

        for (data_bone, label), batch_joint in iterator:
            self.global_step += 1
            data_bone = data_bone.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # Student前向传播
            output = self.model(data_bone)
            loss_ce = self.loss(output, label)

            # Knowledge Distillation Loss
            loss_kd = torch.tensor(0.0).to(self.dev)
            if batch_joint is not None and hasattr(self, 'teacher') and self.teacher:
                data_joint, _ = batch_joint
                data_joint = data_joint.float().to(self.dev, non_blocking=True)

                with torch.no_grad():
                    teacher_logits = self.teacher(data_joint)

                # KL Divergence with Temperature Scaling
                T = 3.0
                loss_kd = F.kl_div(
                    F.log_softmax(output / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T ** 2)

            # Regularization Losses (Entropy + Orthogonality)
            loss_entropy = torch.tensor(0.0).to(self.dev)
            loss_ortho = torch.tensor(0.0).to(self.dev)

            model_core = self.model.module if hasattr(self.model, 'module') else self.model
            for m in model_core.modules():
                if hasattr(m, 'get_loss') and callable(m.get_loss):
                    l_e, l_o = m.get_loss()
                    loss_entropy += l_e
                    loss_ortho += l_o

            # Total Loss
            lambda_kd = getattr(self.arg, 'lambda_kd', 0.0)
            lambda_ent = getattr(self.arg, 'lambda_entropy', 0.0)
            lambda_orth = getattr(self.arg, 'lambda_ortho', 0.0)

            loss = loss_ce + lambda_kd * loss_kd + lambda_ent * loss_entropy + lambda_orth * loss_ortho

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            if hasattr(self.arg, 'grad_clip_norm') and self.arg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip_norm)

            self.optimizer.step()

            # Logging
            self.iter_info['loss'] = loss.item()
            self.iter_info['loss_ce'] = loss_ce.item()
            self.iter_info['loss_kd'] = loss_kd.item()
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
        """
        [FIX] 测试循环
        """
        self.model.eval()

        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            with torch.no_grad():
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
            description='DSA-HGN V5 Recognition with KD'
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

        # Knowledge Distillation
        parser.add_argument('--teacher_weights', type=str, default=None, help='path to teacher model weights')
        parser.add_argument('--lambda_kd', type=float, default=0.3, help='weight for KD loss')

        # Regularization
        parser.add_argument('--lambda_entropy', type=float, default=0.001, help='weight for entropy loss')
        parser.add_argument('--lambda_ortho', type=float, default=0.1, help='weight for orthogonality loss')
        parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='gradient clipping norm')

        # Evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')

        return parser