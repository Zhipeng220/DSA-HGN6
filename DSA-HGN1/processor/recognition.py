import sys
import argparse
import yaml
import math
import numpy as np
import os

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class FT_Processor(Processor):
    """
    Processor for Paper A Supervised Training
    Updated: Uses Entropy Regularization instead of Hard Sparsity.
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)

        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
                eps=1e-4)
        else:
            raise ValueError()

    def adjust_lr(self):
        lr_decay_rate = getattr(self.arg, 'lr_decay_rate', 0.1)

        if hasattr(self.arg, 'warm_up_epoch') and self.arg.warm_up_epoch > 0:
            if self.meta_info['epoch'] < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (self.meta_info['epoch'] + 1) / self.arg.warm_up_epoch
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
                return

        if self.arg.step:
            lr = self.arg.base_lr * (
                    lr_decay_rate ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
            # Export topology when achieving best result
            self.export_topology(self.meta_info['epoch'])

        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def export_topology(self, epoch):
        """
        Exports the learned virtual topology (Adjacency Matrix) for visualization.
        """
        save_path = os.path.join(self.arg.work_dir, f'topology_best_epoch_{epoch}.npy')

        model_core = self.model.module if hasattr(self.model, 'module') else self.model

        target_module = None

        # Search for the first module with 'last_h' (Hypergraph)
        for name, m in model_core.named_modules():
            if hasattr(m, 'last_h') and m.last_h is not None:
                target_module = m
                break

        if target_module is not None:
            # last_h shape: (N, V, M)
            # Take the first sample in the batch
            H = target_module.last_h[0:1]

            # Calculate Virtual Adjacency A = H * H.T (approx)
            # Shape: (1, V, V)
            A_virtual = torch.matmul(H, H.transpose(-1, -2)).squeeze(0).detach().cpu().numpy()

            np.save(save_path, A_virtual)
            # self.io.print_log(f'[VISUALIZATION] Saved virtual topology to: {save_path}')
        else:
            pass

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()

        # [SAFETY GUARD]
        is_multi_stream_model = 'DualBranch' in self.arg.model or 'MultiStream' in self.arg.model
        if is_multi_stream_model:
            if self.arg.stream != 'joint' or self.arg.train_feeder_args.get('bone.yaml', False):
                self.io.print_log(
                    f'[WARNING] Detected {self.arg.model}. Forcing stream="joint" to prevent Double-Diff.')
                self.arg.stream = 'joint'
                self.arg.train_feeder_args['bone.yaml'] = False
                self.arg.train_feeder_args['vel'] = False
                if 'bone.yaml' in self.arg.test_feeder_args: self.arg.test_feeder_args['bone.yaml'] = False
                if 'vel' in self.arg.test_feeder_args: self.arg.test_feeder_args['vel'] = False

        loader = self.data_loader['train']
        loss_value = []
        entropy_loss_value = []
        ortho_loss_value = []

        for data, label in loader:
            self.global_step += 1
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            if self.arg.stream == 'bone.yaml':
                if self.arg.train_feeder_args.get('bone.yaml', False):
                    pass
                else:
                    from net.utils.graph import Graph
                    graph = Graph(self.arg.model_args['graph_args']['layout'])
                    bone = torch.zeros_like(data)
                    for v1, v2 in graph.Bones:
                        bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                    data = bone

            output = self.model(data)

            # 1. Classification Loss
            loss_ce = self.loss(output, label)

            # 2. Regularization Loss (Entropy + Orthogonality)
            loss_entropy = torch.tensor(0.0).to(self.dev)
            loss_ortho = torch.tensor(0.0).to(self.dev)

            model_to_iter = self.model.module if hasattr(self.model, 'module') else self.model

            for m in model_to_iter.modules():
                # Check for get_loss method (DifferentiableSparseHypergraph)
                if hasattr(m, 'get_loss') and callable(m.get_loss):
                    l_e, l_o = m.get_loss()
                    loss_entropy += l_e
                    loss_ortho += l_o

            # Total Loss
            # Using lambda_entropy from config
            loss = loss_ce + (self.arg.lambda_entropy * loss_entropy) + (self.arg.lambda_ortho * loss_ortho)

            self.optimizer.zero_grad()
            loss.backward()

            if self.arg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip_norm)

            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_ce'] = loss_ce.data.item()
            self.iter_info['loss_ent'] = loss_entropy.data.item()
            self.iter_info['loss_orth'] = loss_ortho.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)

            loss_value.append(self.iter_info['loss'])
            entropy_loss_value.append(self.iter_info['loss_ent'])
            ortho_loss_value.append(self.iter_info['loss_orth'])

            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['mean_ent'] = np.mean(entropy_loss_value)
        self.epoch_info['mean_orth'] = np.mean(ortho_loss_value)

        self.show_epoch_info()
        self.train_writer.add_scalar('loss', self.epoch_info['mean_loss'], epoch)
        self.train_writer.add_scalar('loss_entropy', self.epoch_info['mean_ent'], epoch)
        self.train_writer.add_scalar('loss_ortho', self.epoch_info['mean_orth'], epoch)

    def test(self, epoch):
        self.model.eval()

        # [SAFETY GUARD]
        is_multi_stream_model = 'DualBranch' in self.arg.model or 'MultiStream' in self.arg.model
        if is_multi_stream_model:
            if self.arg.stream != 'joint' or self.arg.test_feeder_args.get('bone.yaml', False):
                self.arg.stream = 'joint'
                self.arg.test_feeder_args['bone.yaml'] = False
                self.arg.test_feeder_args['vel'] = False

        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            if self.arg.stream == 'bone.yaml':
                if self.arg.test_feeder_args.get('bone.yaml', False):
                    pass
                else:
                    from net.utils.graph import Graph
                    graph = Graph(self.arg.model_args['graph_args']['layout'])
                    bone = torch.zeros_like(data)
                    for v1, v2 in graph.Bones:
                        bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                    data = bone

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

        for k in self.arg.show_topk:
            self.show_topk(k)

        self.show_best(1)
        self.eval_log_writer(epoch)

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')

        # [Optimization] Regularization Parameters
        # Renamed lambda_sparsity to lambda_entropy for clarity (soft sparsity)
        parser.add_argument('--lambda_entropy', type=float, default=0.001,
                            help='Weight for Entropy loss (Soft Sparsity)')
        parser.add_argument('--lambda_ortho', type=float, default=0.1, help='Weight for Orthogonality loss')

        # Kept for backward compatibility
        parser.add_argument('--lambda_sparsity', type=float, default=0.0, help='Deprecated alias for entropy weight')

        parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping norm')

        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of nearest neighbor mining')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in nearest neighbor mining')

        parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
        parser.add_argument('--warm_up_epoch', type=int, default=0, help='warm up epochs')

        return parser