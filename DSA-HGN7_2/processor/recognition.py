import sys
import argparse
import yaml
import math
import numpy as np
import os
import random
from collections import defaultdict

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


# =============================================================================
# [OPTIMIZER 1] Muon Optimizer (Deep Momentum)
# Based on Nested Learning Paper Sec 2.3
# =============================================================================

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix G.
    This corresponds to the 'non-linear' memory update in Nested Learning.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Use float32 for compatibility with various devices (MPS/CPU)
    X = G.float()

    X /= (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.type_as(G)


class Muon(optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Init momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g + momentum * buf
                else:
                    update = buf

                # [Deep Momentum Logic]
                # Apply Newton-Schulz to orthogonalize the update matrix for 2D+ tensors
                if update.ndim >= 2:
                    original_shape = update.shape
                    # Flatten to 2D: (Out, In * ...)
                    update_flat = update.view(update.size(0), -1)

                    if update_flat.size(0) < update_flat.size(1):
                        # If Out < In, use orthogonalization
                        update_ortho = zeropower_via_newtonschulz5(update_flat, steps=ns_steps)
                    else:
                        # Fallback for skinny matrices or use standard update
                        update_ortho = update_flat

                    # Scale update to have spectral norm similar to original but orthogonalized
                    update_ortho *= max(1, update_flat.size(0) / update_flat.size(1)) ** 0.5

                    update = update_ortho.view(original_shape)

                # Weight update
                p.data.add_(update, alpha=-lr)


# =============================================================================
# [OPTIMIZER 2] Lookahead Optimizer (The Stabilizer)
# Reference: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back"
# =============================================================================

class Lookahead(optim.Optimizer):
    """
    Wraps another optimizer (e.g., Muon or CombinedOptimizer) to improve stability.
    """

    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state

        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            # Interpolate: slow = slow + alpha * (fast - slow)
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
                self.update(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        return {
            "fast_state": fast_state_dict,
            "slow_state": slow_state,
            "param_groups": self.param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = state_dict["fast_state"]
        self.optimizer.load_state_dict(fast_state_dict)
        # Reset Lookahead counter on reload
        for group in self.param_groups:
            group["counter"] = 0


# =============================================================================
# Helper Functions
# =============================================================================

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
    V7.1 Processor: Nested Optimization (Muon + Lookahead) with Label Smoothing
    """

    def load_model(self):
        # 1. Load Student Model
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.model.apply(weights_init)

        # [MODIFIED V7.1] Use Label Smoothing to prevent over-confidence in Muon training
        # This helps with generalization when Training Loss is near zero.
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 2. Load Teacher Model for KD (Knowledge Distillation)
        if self.arg.phase == 'train' and hasattr(self.arg, 'teacher_weights') and self.arg.teacher_weights:
            self.io.print_log(f'[KD] Loading Teacher from: {self.arg.teacher_weights}')

            teacher_args = self.arg.model_args.copy()
            teacher_args['in_channels'] = 3
            teacher_args['use_physical'] = False

            self.teacher = self.io.load_model(self.arg.model, **teacher_args)

            weights = torch.load(self.arg.teacher_weights, map_location='cpu')
            if isinstance(weights, dict):
                if 'model' in weights:
                    weights = weights['model']
                elif 'state_dict' in weights:
                    weights = weights['state_dict']

            from collections import OrderedDict
            new_weights = OrderedDict()
            for k, v in weights.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_weights[name] = v

            self.teacher.load_state_dict(new_weights, strict=False)

            if self.arg.use_gpu:
                self.teacher = self.teacher.to(self.dev)

            self.teacher.eval()

            # Ensure Teacher BN buffers are on device
            for module in self.teacher.modules():
                if isinstance(module, torch.nn.BatchNorm1d) or \
                        isinstance(module, torch.nn.BatchNorm2d) or \
                        isinstance(module, torch.nn.BatchNorm3d):
                    if module.running_mean is not None:
                        module.running_mean = module.running_mean.to(self.dev)
                    if module.running_var is not None:
                        module.running_var = module.running_var.to(self.dev)
                    if module.num_batches_tracked is not None:
                        module.num_batches_tracked = module.num_batches_tracked.to(self.dev)

            # Freeze Teacher
            for param in self.teacher.parameters():
                param.requires_grad = False

            self.io.print_log('[KD] Teacher loaded successfully.')
        else:
            self.teacher = None

    def load_optimizer(self):
        """
        [MODIFIED] Setup Muon + AdamW, then wrap with Lookahead
        """
        if self.arg.optimizer == 'Muon':
            self.io.print_log('[Optimizer] Initializing Muon (Deep Momentum) + AdamW...')

            muon_params = []
            adamw_params = []

            # Split parameters: 2D+ for Muon, 1D/Scalar for AdamW
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim >= 2:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

            # Inner Optimizer 1: Muon
            self.optimizer_muon = Muon(
                muon_params,
                lr=0.02,  # Default Muon LR
                momentum=0.95
            )
            # Inner Optimizer 2: AdamW
            self.optimizer_adamw = optim.AdamW(
                adamw_params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
                eps=1e-4
            )

            # Inner Wrapper: Combine them
            class CombinedOptimizer:
                def __init__(self, opts):
                    self.opts = opts

                def zero_grad(self):
                    for opt in self.opts: opt.zero_grad()

                def step(self, closure=None):
                    for opt in self.opts: opt.step()

                def state_dict(self):
                    return [opt.state_dict() for opt in self.opts]

                def load_state_dict(self, states):
                    for opt, state in zip(self.opts, states): opt.load_state_dict(state)

                @property
                def param_groups(self):
                    # Return combined groups for LR scheduler
                    return self.opts[0].param_groups + self.opts[1].param_groups

                @property
                def state(self):
                    # Placeholder for Lookahead access
                    return {}

            self.base_optimizer = CombinedOptimizer([self.optimizer_muon, self.optimizer_adamw])

            # [CRITICAL] Outer Wrapper: Lookahead
            # k=5 steps forward, 0.5 interpolation
            self.optimizer = Lookahead(self.base_optimizer, k=5, alpha=0.5)
            self.io.print_log(
                f'[Optimizer] Nested Optimization Enabled: Lookahead(Muon: {len(muon_params)} params + AdamW: {len(adamw_params)} params)')

        elif self.arg.optimizer == 'SGD':
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
        self.data_loader = dict()
        train_mean_map = None
        train_std_map = None

        # 1. Train Loader
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

            # Get stats
            if hasattr(self.data_loader['train'].dataset, 'mean_map'):
                train_mean_map = self.data_loader['train'].dataset.mean_map
                train_std_map = self.data_loader['train'].dataset.std_map
                self.io.print_log('[DATA] Train stats loaded from Bone stream.')

            # 2. Joint Loader for KD
            if self.arg.phase == 'train' and hasattr(self.arg, 'teacher_weights') and self.arg.teacher_weights:
                joint_args = self.arg.train_feeder_args.copy()
                joint_args['bone'] = False
                joint_args['vel'] = False
                joint_args['normalization'] = True
                joint_args['random_rot'] = False

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
                self.io.print_log('[KD] Joint stream loader created for Distillation.')

        # 3. Test Loader
        if self.arg.test_feeder_args:
            test_feeder = import_class(self.arg.test_feeder)
            test_args = self.arg.test_feeder_args.copy()

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
        self.model.train()
        if hasattr(self, 'teacher') and self.teacher:
            self.teacher.eval()

        self.adjust_lr()

        loader_bone = self.data_loader['train']
        loader_joint = self.data_loader.get('train_joint', None)

        # Zip loaders if Joint stream is available for KD
        if loader_joint and hasattr(self, 'teacher') and self.teacher:
            iterator = zip(loader_bone, loader_joint)
        else:
            iterator = ((batch, None) for batch in loader_bone)

        loss_value = []

        # Debug info for first batch
        first_batch = True

        for (data_bone, label), batch_joint in iterator:
            self.global_step += 1
            data_bone = data_bone.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            if first_batch and batch_joint is not None:
                # Optional debug print for data ranges can go here
                first_batch = False

            # Forward
            output = self.model(data_bone)
            loss_ce = self.loss(output, label)

            # KD Loss
            loss_kd = torch.tensor(0.0).to(self.dev)
            if batch_joint is not None and hasattr(self, 'teacher') and self.teacher:
                data_joint, _ = batch_joint
                data_joint = data_joint.float().to(self.dev, non_blocking=True)

                with torch.no_grad():
                    teacher_logits = self.teacher(data_joint)

                T = 4.0
                loss_kd = F.kl_div(
                    F.log_softmax(output / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T ** 2)

            # Regularization Losses
            loss_entropy = torch.tensor(0.0).to(self.dev)
            loss_ortho = torch.tensor(0.0).to(self.dev)

            model_core = self.model.module if hasattr(self.model, 'module') else self.model
            for m in model_core.modules():
                if hasattr(m, 'get_loss') and callable(m.get_loss):
                    l_e, l_o = m.get_loss()
                    loss_entropy += l_e
                    loss_ortho += l_o

            lambda_kd = getattr(self.arg, 'lambda_kd', 0.0)
            lambda_ent = getattr(self.arg, 'lambda_entropy', 0.0)
            lambda_orth = getattr(self.arg, 'lambda_ortho', 0.0)

            loss = loss_ce + lambda_kd * loss_kd + lambda_ent * loss_entropy + lambda_orth * loss_ortho

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

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
                # Update all internal groups
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
            description='DSA-HGN V7 Recognition with Lookahead + Muon'
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