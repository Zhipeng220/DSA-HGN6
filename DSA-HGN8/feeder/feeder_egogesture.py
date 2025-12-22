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
    EgoGesture/SHREC Dataset Feeder for DSA-HGN V5
    [V5修复] 正确处理Bone/Velocity统计量计算和外部注入
    [致命修复] 解决KD训练时Joint流和Bone流的RNG不同步问题
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
                 mean_map=None,  # [FIX] 外部注入的均值
                 std_map=None,  # [FIX] 外部注入的方差
                 repeat=1):  # [FIX] 数据重复次数

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
        self.repeat = repeat

        # [FIX] 初始化统计量
        self.mean_map = mean_map
        self.std_map = std_map

        self.load_data()

        # [FIX] 只有当外部没有传入统计量时(通常是训练集初始化时),才计算新的统计量
        if normalization:
            if self.mean_map is None or self.std_map is None:
                self.get_mean_map()
            else:
                print(f"[Feeder] Using external statistics (mean/std injected)")

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
            with open(self.label_path, 'rb') as f:
                self.label, self.sample_name = pickle.load(f)

        # Debug mode
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        print(f"[Feeder] Data shape: {self.data.shape}, Samples: {len(self.label)}")

    def get_mean_map(self):
        """
        [FIX] 修复统计量计算逻辑
        关键修复:先转换数据流(Bone/Velocity),再计算统计量
        """
        data = self.data
        N, C, T, V, M = data.shape

        # [FIX 1] Bone Motion 流需要同时应用 bone 和 vel
        # 处理顺序：Joint -> Bone -> Bone Velocity

        # [FIX 2] 先应用 Bone 转换（如果需要）
        if self.bone:
            stream_name = "Bone Motion" if self.vel else "Bone"
            print(f"[Feeder] Calculating mean/std for {stream_name} stream...")
            bone_data = np.zeros_like(data)
            for v1, v2 in self.get_bone_connections():
                bone_data[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
            data = bone_data

        # [FIX 3] 再应用 Velocity 转换（如果需要）
        # 注意：这里的 data 可能是 Bone 向量（如果 self.bone=True）
        if self.vel:
            if not self.bone:
                print("[Feeder] Calculating mean/std for Joint Motion stream...")
            print("[Feeder] Calculating mean/std for Velocity stream...")
            vel_data = np.zeros_like(data)
            vel_data[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
            vel_data[:, :, -1, :, :] = 0  # 最后一帧速度补0
            data = vel_data

        # 计算统计量
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)

        # [FIX 4] 增加 1e-4 防止除以零 (对 MPS/Float16 非常重要)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape(
            (C, 1, V, 1)) + 1e-4

        print(f"[Feeder] Stats computed. Mean shape: {self.mean_map.shape}, Std shape: {self.std_map.shape}")

    def __len__(self):
        # [FIX] 乘以repeat次数
        return len(self.label) * self.repeat

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # [FIX] 索引取模,防止越界
        index = index % len(self.label)

        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # =====================================================================
        # [STEP 1] 空间增强(在数据流转换之前)
        # =====================================================================

        # [致命修复] 解决KD训练时的RNG不同步问题
        # 策略: 无论是否应用增强,都消耗相同的随机数,确保Joint流和Bone流完全同步

        if self.random_move:
            # 生成随机平移参数(这会消耗RNG状态)
            move_params = self._generate_random_move_params(data_numpy)

            # 只有非Bone流才真正应用平移
            # Bone流虽然不应用,但RNG状态已经消耗,保证与Joint流同步
            if not self.bone:
                data_numpy = self._apply_random_move(data_numpy, move_params)
            # else: Bone流跳过应用,但RNG已同步消耗

        # [优化 2] Random rotation对所有流都有效且物理正确
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        # [优化 3] Shear增强
        if self.split == 'train' and self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        # =====================================================================
        # [STEP 2] 数据流转换 (Position -> Velocity / Bone)
        # =====================================================================

        # [STEP 2.1] Bone 转换（如果需要）
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in self.get_bone_connections():
                bone_data_numpy[:, :, v1, :] = data_numpy[:, :, v1, :] - data_numpy[:, :, v2, :]
            data_numpy = bone_data_numpy

        # [STEP 2.2] Velocity 转换（如果需要）
        # 注意：此时 data_numpy 可能已经是 Bone 向量
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        # =====================================================================
        # [STEP 3] 时间增强 & 归一化
        # =====================================================================

        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_padding(data_numpy, self.window_size)

        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)

        if self.split == 'train' and self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        # [FIX] 归一化时确保除数不为0
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / (self.std_map + 1e-4)
            data_numpy = np.nan_to_num(data_numpy, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)

        return data_numpy, label

    def _generate_random_move_params(self, data_numpy):
        """
        [新增方法] 生成随机平移参数(消耗RNG状态)
        这确保了即使不应用平移,RNG状态也被同步消耗
        """
        C, T, V, M = data_numpy.shape
        # 生成3个随机平移值(XYZ方向)
        move_params = np.array([
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ])
        return move_params

    def _apply_random_move(self, data_numpy, move_params):
        """
        [新增方法] 应用随机平移
        将平移参数的生成和应用分离,确保RNG同步
        """
        C, T, V, M = data_numpy.shape
        data_numpy[:C, :, :, :] += move_params.reshape(C, 1, 1, 1)
        return data_numpy

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def get_bone_connections(self):
        """
        定义骨骼连接关系
        [FIX 3] 自动检测关节数,支持多种数据集
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
        elif num_joints == 21:
            # EgoGesture (21 Joints)
            connections = [
                (0, 0), (1, 0), (2, 1), (3, 2), (4, 3),
                (5, 0), (6, 5), (7, 6), (8, 7),
                (9, 0), (10, 9), (11, 10), (12, 11),
                (13, 0), (14, 13), (15, 14), (16, 15),
                (17, 0), (18, 17), (19, 18), (20, 19)
            ]
        elif num_joints == 25:
            # NTU RGB+D (25 Joints) - 示例拓扑
            connections = [(i, max(0, i - 1)) for i in range(num_joints)]
            print(f"[Warning] Using default topology for {num_joints} joints")
        else:
            # 通用回退: 每个关节连接到前一个关节
            connections = [(i, max(0, i - 1)) for i in range(num_joints)]
            print(f"[Warning] Unknown joint count {num_joints}, using linear topology")

        return connections


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod