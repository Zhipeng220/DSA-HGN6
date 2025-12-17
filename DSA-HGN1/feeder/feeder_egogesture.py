import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import random

sys.path.extend(['../'])
from feeder import tools


class Feeder(Dataset):
    """
    EgoGesture Dataset Feeder for AimCLR
    Arguments:
        data_path: path to data file (.npy)
        label_path: path to label file (.pkl)
        split: 'train' or 'test'
        random_choose: randomly choose a portion of the input sequence
        random_shift: randomly pad zeros at the begining or end of sequence
        random_move: randomly move the sequence
        window_size: The length of the output sequence
        normalization: normalize input sequence
        debug: only use first 100 samples for debugging
        use_mmap: use memory map to load data
        bone: use bone.yaml stream instead of joint stream
        vel: use velocity stream
        random_rot: randomly rotate the skeleton
        p_interval: sampling interval for test
        shear_amplitude: amplitude of shear augmentation
        temperal_padding_ratio: temporal padding ratio for augmentation
        mean_map: (New) external mean map for normalization
        std_map: (New) external std map for normalization
        repeat: (New) number of times to repeat the dataset
    """

    def __init__(self,
                 data_path,
                 label_path,
                 split='train',
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 normalization=False,
                 debug=False,
                 use_mmap=True,
                 bone=False,
                 vel=False,
                 random_rot=False,
                 p_interval=1,
                 shear_amplitude=0.5,
                 temperal_padding_ratio=6,
                 mean_map=None,  # [FIX] 新增：接收外部传入的均值
                 std_map=None,  # [FIX] 新增：接收外部传入的方差
                 repeat=1):  # [FIX] 新增：repeat 参数

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.bone = bone
        self.vel = vel
        self.random_rot = random_rot
        self.p_interval = p_interval
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.repeat = repeat  # [FIX] 保存 repeat 参数

        # [FIX] 初始化统计量
        self.mean_map = mean_map
        self.std_map = std_map

        self.load_data()

        if normalization:
            # [FIX] 只有当外部没有传入统计量时（通常是训练集初始化时），才计算新的统计量
            if self.mean_map is None or self.std_map is None:
                self.get_mean_map()

    def load_data(self):
        # Load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # Load label
        try:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # Handle different pickle formats
            with open(self.label_path, 'rb') as f:
                self.label, self.sample_name = pickle.load(f)

        # Debug mode: only use first 100 samples
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        print(f"Data shape: {self.data.shape}")
        print(f"Number of samples: {len(self.label)}")

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape

        # [FIX 1] 如果是 Bone 模式，先将数据转换为 Bone 向量再计算统计量
        if self.bone:
            print("Calculating mean/std for Bone stream...")
            bone_data = np.zeros_like(data)
            for v1, v2 in self.get_bone_connections():
                bone_data[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
            data = bone_data

        # [FIX 2 - 关键修复] 如果是 Velocity (Motion) 模式，先计算速度再算均值！
        # 之前的代码直接用位置(data)的均值去归一化速度，导致数值完全错误。
        if self.vel:
            print("Calculating mean/std for Velocity stream...")
            vel_data = np.zeros_like(data)
            vel_data[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
            # 最后一帧速度补0
            vel_data[:, :, -1, :, :] = 0
            data = vel_data

        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)

        # [FIX 3] 增加 1e-4 防止除以零 (对 MPS/Float16 非常重要)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape(
            (C, 1, V, 1)) + 1e-4

    def __len__(self):
        # [FIX] 乘以 repeat 次数
        return len(self.label) * self.repeat

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # [FIX] 索引取模，防止越界
        index = index % len(self.label)

        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # ---------------------------------------------------------------------
        # [STEP 1] 空间增强
        # ---------------------------------------------------------------------

        # [优化 1] 仅对非 Bone 流或确实需要平移的任务启用 random_move
        # Bone 流的平移会被抵消，做也是白做，不如跳过
        if self.random_move and not self.bone:
            data_numpy = tools.random_move(data_numpy)

        # [优化 2] Random rotation 对所有流都有效且物理正确
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        # [优化 3] Shear 增强
        if self.split == 'train' and self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        # ---------------------------------------------------------------------
        # [STEP 2] 数据流转换 (Position -> Velocity / Bone)
        # ---------------------------------------------------------------------
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in self.get_bone_connections():
                bone_data_numpy[:, :, v1, :] = data_numpy[:, :, v1, :] - data_numpy[:, :, v2, :]
            data_numpy = bone_data_numpy

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        # ---------------------------------------------------------------------
        # [STEP 3] 时间增强 & 归一化 (保持不变)
        # ---------------------------------------------------------------------
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_padding(data_numpy, self.window_size)

        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)

        if self.split == 'train' and self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.normalization:
            # 确保除数不为0
            data_numpy = (data_numpy - self.mean_map) / (self.std_map + 1e-4)
            data_numpy = np.nan_to_num(data_numpy, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def get_bone_connections(self):
        """
        Define bone.yaml connections for hand skeleton (22 joints for SHREC)
        """
        num_joints = self.data.shape[3]

        if num_joints == 22:
            # SHREC'17 Track Layout (22 Joints)
            connections = [
                (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4),
                (6, 1), (7, 6), (8, 7), (9, 8),
                (10, 1), (11, 10), (12, 11), (13, 12),
                (14, 1), (15, 14), (16, 15), (17, 16),
                (18, 1), (19, 18), (20, 19), (21, 20)
            ]
        else:
            # EgoGesture (21 Joints)
            connections = [
                (0, 0), (1, 0), (2, 1), (3, 2), (4, 3),
                (5, 0), (6, 5), (7, 6), (8, 7),
                (9, 0), (10, 9), (11, 10), (12, 11),
                (13, 0), (14, 13), (15, 14), (16, 15),
                (17, 0), (18, 17), (19, 18), (20, 19)
            ]

        return connections


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod