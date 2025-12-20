# DSA-HGN: 用于骨骼动作识别的动态稀疏自适应超图网络

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **注意**: 本项目目前处于实验阶段,结果和实现可能会随开发进展而更新。

## 📋 目录

- [项目概述](#-项目概述)
- [核心特性](#-核心特性)
- [模型架构](#️-模型架构)
- [环境安装](#-环境安装)
- [数据集准备](#-数据集准备)
- [四流训练指南](#-四流训练指南)
  - [Joint 流 (关节)](#1-训练-joint-流关节)
  - [Bone 流 (骨骼)](#2-训练-bone-流骨骼)
  - [J-Motion 流 (关节运动)](#3-训练-j-motion-流关节运动)
  - [B-Motion 流 (骨骼运动)](#4-训练-b-motion-流骨骼运动)
- [模型评估](#-模型评估)
- [多流融合](#-多流融合)
- [性能分析工具](#-性能分析工具)
- [模型库](#-模型库)
- [项目结构](#-项目结构)
- [配置参数说明](#-配置参数说明)
- [故障排除](#-故障排除)
- [致谢](#-致谢)

## 🎯 项目概述

DSA-HGN 是一个专为骨骼动作识别任务设计的新型深度学习框架。它引入了**动态稀疏自适应超图网络 (Dynamic Sparse Adaptive Hypergraph Network)**,利用带有熵正则化软稀疏性的超图卷积,来对人体骨骼数据中复杂的关节关系进行建模。

### 支持的数据集

- **EgoGesture**: 以自我为中心的手势识别 (83 类, 21 个关节)
- **SHREC'17 Track**: 3D 手势识别 (14 类, 22 个关节)

### 多流架构

框架支持四种互补的数据流:

- **Joint Stream (关节流)**: 原始关节坐标
- **Bone Stream (骨骼流)**: 连接关节之间的骨骼向量
- **Joint-Motion Stream (关节运动流)**: 关节的时间差分(速度)
- **Bone-Motion Stream (骨骼运动流)**: 骨骼向量的时间差分

## ✨ 核心特性

### 1. 动态稀疏超图模块

- **熵正则化 Softmax**: 替代硬性的 Top-K 选择,实现可微的软稀疏性
- **可学习原型**: 正交初始化的超边原型
- **梯度流**: 确保训练期间所有原型都能接收到梯度

### 2. 双分支架构

- **时空分支**: 捕捉标准的 ST-GCN 模式
- **通道微分分支**: 建模通道间的关系

### 3. 超图注意力融合模块 (HAFM)

- 多流的自适应加权
- 端到端可学习的融合策略

### 4. 硬件兼容性

- 原生支持 **Apple Silicon (MPS)** 后端
- CUDA 和 CPU 回退支持
- 支持混合精度训练

## 🏗️ 模型架构

```
输入骨骼序列 (N, C, T, V, M)
         ↓
    数据 BN 层
         ↓
┌────────────────────────┐
│   10层 ST-GCN          │
│   结合超图卷积模块      │
└────────────────────────┘
         ↓
    全局平均池化
         ↓
     Dropout 层
         ↓
 全连接分类器 (num_classes)
```

### 超图卷积单元

```
节点特征 (N, C, T, V)
         ↓
    查询投影
         ↓
    原型匹配
         ↓
熵正则化 Softmax → 关联矩阵 H (N, V, M)
         ↓
  V2E 聚合 (H @ X)
         ↓
    边卷积
         ↓
  E2V 传播 (H^T @ E)
         ↓
   残差 + BN + ReLU
```

## 🔧 环境安装

### 前置要求

```bash
Python >= 3.8
PyTorch >= 1.10.0
CUDA >= 11.1 (GPU训练可选)
```

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/DSA-HGN.git
cd DSA-HGN

# 2. 创建 conda 环境
conda create -n dsa_hgn python=3.8
conda activate dsa_hgn

# 3. 安装 PyTorch (示例: CUDA 11.3)
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia

# Apple Silicon (M1/M2/M3) 用户
# 最新版本的 PyTorch 自动支持 MPS

# 4. 安装依赖
pip install -r requirements.txt

# 5. 安装 torchlight 模块
cd torchlight
python setup.py install
cd ..
```

### 依赖包

```txt
numpy>=1.19.0
pyyaml>=5.4.0
tensorboardX>=2.4.0
h5py>=3.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
networkx>=2.5.0
tqdm>=4.60.0
```

## 📊 数据集准备

### SHREC'17 Track 数据集

1. **下载数据集**:
   ```bash
   # 从官方源下载
   # http://www-rech.telecom-lille.fr/shrec2017-hand/
   ```

2. **数据目录结构**:
   ```
   DATA/
   └── SHREC2017_data/
       ├── train_data.npy      # 形状: (N_train, C, T, V, M)
       ├── train_label.pkl     # 列表: [sample_names, labels]
       ├── val_data.npy        # 形状: (N_val, C, T, V, M)
       └── val_label.pkl
   ```

3. **更新配置文件路径**:
   ```yaml
   # config/SHREC/joint/joint.yaml
   train_feeder_args:
     data_path: /path/to/DATA/SHREC2017_data/train_data.npy
     label_path: /path/to/DATA/SHREC2017_data/train_label.pkl
   
   test_feeder_args:
     data_path: /path/to/DATA/SHREC2017_data/val_data.npy
     label_path: /path/to/DATA/SHREC2017_data/val_label.pkl
   ```

### EgoGesture 数据集

1. **下载并提取**:
   ```bash
   # 遵循 CTR-GCN 预处理流程
   # https://github.com/Uason-Chen/CTR-GCN
   ```

2. **数据目录结构**:
   ```
   data/
   └── egogesture/
       ├── train_data.npy
       ├── train_label.pkl
       ├── val_data.npy
       └── val_label.pkl
   ```

### 数据格式说明

**NumPy 数组格式** (`.npy`):
```python
形状: (N, C, T, V, M)
# N: 样本数量
# C: 通道数 (通常为 3,表示 x, y, z 坐标)
# T: 时序长度 (帧数)
# V: 关节数量 (EgoGesture: 21, SHREC: 22)
# M: 人数 (手势识别通常为 1)
```

**标签格式** (`.pkl`):
```python
[sample_names, labels]
# sample_names: 字符串标识符列表
# labels: 整数类别标签列表
```

## 🚀 四流训练指南

我们需要分别训练四个独立的模型。建议为每个流指定不同的 `work_dir` 以免覆盖结果。

### 1. 训练 Joint 流(关节)

使用原始关节坐标数据:

```bash
python main.py finetune_evaluation \
    --config config/SHREC/joint/joint.yaml \
    --work_dir work_dir/SHREC/joint \
    --device 0 \
    --batch_size 32 \
    --num_epoch 60
```

### 2. 训练 Bone 流(骨骼)

计算连接关节之间的骨骼向量:

```bash
python main.py finetune_evaluation \
    --config config/SHREC/bone.yaml/bone.yaml.yaml \
    --work_dir work_dir/SHREC/bone.yaml \
    --device 0 \
    --batch_size 32 \
    --num_epoch 60
```

### 3. 训练 J-Motion 流(关节运动)

计算关节的时间差分(速度):

```bash
python main.py finetune_evaluation \
    --config config/SHREC/Jmotion/jmotion.yaml \
    --work_dir work_dir/SHREC/jmotion \
    --device 0 \
    --batch_size 32 \
    --num_epoch 60
```

### 4. 训练 B-Motion 流(骨骼运动)

计算骨骼向量的时间差分:

```bash
python main.py finetune_evaluation \
    --config config/SHREC/Bmotion/bmotion.yaml \
    --work_dir work_dir/SHREC/bmotion \
    --device 0 \
    --batch_size 32 \
    --num_epoch 60
```

### 从检查点恢复训练

```bash
python main.py finetune_evaluation \
    --config config/SHREC/bone.yaml/bone.yaml.yaml \
    --weights work_dir/SHREC/bone.yaml/epoch025_acc87.14_model.pt \
    --start_epoch 25
```

### 多流融合训练

```bash
python main.py finetune_evaluation \
    --config config/SHREC/fusion/hafm_fusion.yaml \
    --work_dir work_dir/SHREC/hafm_fusion \
    --device 0 \
    --batch_size 16  # 由于内存需求,减小批次大小
```

### Apple Silicon 上训练

```bash
# MPS 后端自动检测
python main.py finetune_evaluation \
    --config config/SHREC/joint/joint.yaml \
    --device 0 \
    --use_gpu True
```

### 关键训练参数

| 参数 | 描述 | 默认值 | 建议值 |
|------|------|--------|--------|
| `base_lr` | 初始学习率 | 0.05 | SGD: 0.05, AdamW: 0.001 |
| `num_epoch` | 总训练轮数 | 60 | 60-150 |
| `batch_size` | 每GPU批次大小 | 32 | 单流: 32, 融合: 16 |
| `lambda_entropy` | 熵正则化权重 | 0.001 | 0.001-0.005 |
| `lambda_ortho` | 正交损失权重 | 0.1 | 0.1 |
| `grad_clip_norm` | 梯度裁剪阈值 | 1.0 | 1.0 |
| `num_hyperedges` | 超图边数量 | 16 | 16 |

## 🧪 模型评估

### 生成推理结果

训练完成后,使用表现最好的模型权重生成用于融合的结果文件:

#### 1. 生成 Joint 流结果

```bash
python main.py finetune_evaluation \
    --config config/SHREC/joint/joint.yaml \
    --phase test \
    --weights work_dir/SHREC/joint/best_model.pt \
    --save_result True \
    --work_dir work_dir/SHREC/joint
```

输出: `work_dir/SHREC/joint/test_result.pkl`

#### 2. 生成 Bone 流结果

```bash
python main.py finetune_evaluation \
    --config config/SHREC/bone.yaml/bone.yaml.yaml \
    --phase test \
    --weights work_dir/SHREC/bone.yaml/best_model.pt \
    --save_result True \
    --work_dir work_dir/SHREC/bone.yaml
```

#### 3. 生成 J-Motion 流结果

```bash
python main.py finetune_evaluation \
    --config config/SHREC/Jmotion/jmotion.yaml \
    --phase test \
    --weights work_dir/SHREC/jmotion/best_model.pt \
    --save_result True \
    --work_dir work_dir/SHREC/jmotion
```

#### 4. 生成 B-Motion 流结果

```bash
python main.py finetune_evaluation \
    --config config/SHREC/Bmotion/bmotion.yaml \
    --phase test \
    --weights work_dir/SHREC/bmotion/best_model.pt \
    --save_result True \
    --work_dir work_dir/SHREC/bmotion
```

## 🔗 多流融合

为了获得最佳性能,我们将四个流的结果进行加权融合。

### 四流融合 (SHREC)

修改并运行 `ensemble_shrec.py`:

```python
import pickle
import numpy as np
from tqdm import tqdm

# ========== 路径配置 ==========
joint_path   = 'work_dir/SHREC/joint/test_result.pkl'
bone_path    = 'work_dir/SHREC/bone.yaml/test_result.pkl'
jmotion_path = 'work_dir/SHREC/jmotion/test_result.pkl'
bmotion_path = 'work_dir/SHREC/bmotion/test_result.pkl'
label_path   = '/path/to/SHREC2017_data/val_label.pkl'

# ========== 融合权重 [Joint, Bone, J-Motion, B-Motion] ==========
# 推荐策略:
# - 均衡: [1.0, 1.0, 1.0, 1.0]
# - SOTA常见: [1.0, 1.0, 0.6, 0.6] (适度降低运动流权重)
alpha = [1.0, 1.0, 0.5, 0.5]

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# 加载所有流的预测结果
r1 = load_pkl(joint_path)
r2 = load_pkl(bone_path)
r3 = load_pkl(jmotion_path)
r4 = load_pkl(bmotion_path)

# 加载真实标签
with open(label_path, 'rb') as f:
    label_data = pickle.load(f)
    sample_names, true_labels = label_data[0], label_data[1]

right_num = 0
total_num = 0

for i in tqdm(range(len(sample_names))):
    name = sample_names[i]
    label = int(true_labels[i])
    
    if name not in r1:
        continue
    
    # 加权融合四个流的预测分数
    score = (r1[name] * alpha[0]) + (r2[name] * alpha[1]) + \
            (r3[name] * alpha[2]) + (r4[name] * alpha[3])
    
    # 预测类别
    if np.argmax(score) == label:
        right_num += 1
    total_num += 1

accuracy = right_num / total_num * 100
print(f'四流融合准确率: {accuracy:.2f}%')
```

运行融合脚本:

```bash
python ensemble_shrec.py
```

### 双流融合 (EgoGesture)

```bash
python ensemble_egogesture.py
```

配置示例:

```python
# 路径配置
joint_path = 'work_dir/egogesture/joint/test_result.pkl'
bone_path = 'work_dir/egogesture/bone.yaml/test_result.pkl'
label_path = '/path/to/egogesture/val_label.pkl'

# 融合权重 [Joint, Bone]
alpha = [0.5, 0.5]  # 等权重
```

## 📈 性能分析工具

### 混淆矩阵可视化

```bash
python tools/Confusion\ Matrix.py
```

生成 `SHREC_Confusion_Matrix.png`,展示每个类别的预测分布。

### 错误分析

```bash
python tools/Error\ Analysis.py
```

输出内容:
- Top-5 混淆对
- 每类错误率
- 误分类模式

### 拓扑可视化

```bash
python tools/visualize_topology.py \
    work_dir/SHREC/joint/topology_best_epoch_50.npy \
    --threshold 0.1
```

可视化学习到的超图虚拟连接。

## 📦 模型库

> **注意**: 模型目前处于实验阶段。预训练权重将在论文接收后发布。

预期性能(可能变化):

| 数据集 | 流类型 | 训练轮数 | Top-1 准确率 | 配置文件 |
|--------|--------|----------|--------------|----------|
| SHREC'17 | Joint | 60 | ~85% | `config/SHREC/joint/joint.yaml` |
| SHREC'17 | Bone | 60 | ~87% | `config/SHREC/bone/bone.yaml` |
| SHREC'17 | J-Motion | 60 | ~82% | `config/SHREC/Jmotion/jmotion.yaml` |
| SHREC'17 | B-Motion | 60 | ~84% | `config/SHREC/Bmotion/bmotion.yaml` |
| SHREC'17 | 四流融合 | - | ~95% | - |
| EgoGesture | Joint | 60 | 待定 | `config/egogesture/supervised/hyperhand_supervised.yaml` |

## 📁 项目结构

```
DSA-HGN/
├── config/                          # 配置文件
│   ├── SHREC/
│   │   ├── joint/                   # Joint流配置
│   │   ├── bone/                    # Bone流配置
│   │   ├── Jmotion/                 # J-Motion流配置
│   │   ├── Bmotion/                 # B-Motion流配置
│   │   └── fusion/                  # 多流融合配置
│   └── egogesture/
│       └── supervised/
├── feeder/                          # 数据加载与增强
│   ├── feeder_egogesture.py        # 主数据加载器
│   └── tools.py                     # 数据增强函数
├── graph/                           # 图拓扑定义
│   ├── shrec.py                     # SHREC骨架图
│   ├── egogesture.py               # EgoGesture骨架图
│   └── tools.py                     # 图工具
├── net/                             # 网络架构
│   ├── dsa_hgn.py                  # 主模型
│   ├── hypergraph_modules.py       # 超图卷积层
│   ├── basic_modules.py            # GCN和TCN模块
│   └── utils/                       # 网络工具
├── processor/                       # 训练与评估逻辑
│   ├── processor.py                # 基础处理器类
│   ├── recognition.py              # 识别处理器
│   └── io.py                        # I/O操作
├── tools/                           # 分析与可视化工具
│   ├── Confusion Matrix.py         # 混淆矩阵生成
│   ├── Error Analysis.py           # 错误模式分析
│   └── visualize_topology.py       # 拓扑可视化
├── torchlight/                      # 训练工具
│   └── io.py                        # 模型I/O与日志
├── ensemble_shrec.py               # SHREC融合评估
├── ensemble_egogesture.py          # EgoGesture融合评估
├── main.py                          # 主入口
└── README.md                        # 本文件
```

## ⚙️ 配置参数说明

### 模型架构

```yaml
model_args:
  in_channels: 3                     # 输入通道数 (x, y, z)
  base_channels: 64                  # 基础特征维度
  num_stages: 10                     # ST-GCN层数
  inflate_stages: [5, 8]            # 通道翻倍的层
  down_stages: [5, 8]               # 时序下采样的层
  num_hyperedges: 16                # 超图原型数量
  adaptive: true                     # 启用自适应图学习
  use_virtual_conn: True            # 启用超图连接
  drop_out: 0.0                      # Dropout率
```

### 数据增强

```yaml
train_feeder_args:
  window_size: 180                   # 时序窗口长度
  normalization: False               # 应用z-score归一化
  random_choose: True               # 随机时序裁剪
  random_shift: True                # 随机时序位移
  random_rot: True                  # 随机旋转增强
  shear_amplitude: 0.5              # 剪切变换强度
  temperal_padding_ratio: 6         # 时序填充比率
  repeat: 5                          # 数据集重复因子
```

### 训练策略

```yaml
optimizer: SGD                       # 优化器 (SGD/Adam/AdamW)
base_lr: 0.05                        # 初始学习率
weight_decay: 0.0005                # L2正则化
nesterov: True                       # 使用Nesterov动量
grad_clip_norm: 1.0                 # 梯度裁剪阈值

# 学习率调度
step: [30, 50]                      # LR衰减里程碑
lr_decay_rate: 0.1                  # LR衰减因子
warm_up_epoch: 5                    # 预热轮数

# 正则化
lambda_entropy: 0.001               # 软稀疏权重
lambda_ortho: 0.1                   # 原型正交性权重
```

## 🛠️ 故障排除

### 常见问题

1. **CUDA 显存不足**:
   ```bash
   # 减小批次大小
   --batch_size 16
   # 或使用梯度累积 (需修改 processor.py)
   ```

2. **MPS 后端问题 (Mac)**:
   ```bash
   # 如果MPS失败,强制使用CPU模式
   --use_gpu False
   ```

3. **数据加载错误**:
   ```python
   # 检查配置文件中的数据路径
   # 确保 .npy 和 .pkl 文件存在
   # 验证数据形状: (N, C, T, V, M)
   ```

4. **训练中出现 NaN 损失**:
   ```yaml
   # 降低学习率
   base_lr: 0.01  # 而不是 0.05
   
   # 启用梯度裁剪
   grad_clip_norm: 1.0
   
   # 增加熵权重
   lambda_entropy: 0.005
   ```

### 调试模式

```bash
# 使用减小的数据集快速测试
python main.py finetune_evaluation \
    --config config/SHREC/joint/joint.yaml \
    --debug True \
    --num_epoch 2
```

## 📊 训练监控

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir work_dir/SHREC/joint

# 在浏览器中查看: http://localhost:6006
```

**可用指标**:
- 训练损失 (交叉熵 + 熵 + 正交性)
- 学习率调度
- 验证准确率 (Top-1, Top-5)
- 每轮统计数据

### 日志文件

```bash
# 查看训练日志
tail -f work_dir/SHREC/joint/log.txt

# 检查保存的模型
ls work_dir/SHREC/joint/*.pt
```

## 🔍 超参数调优

### 学习率搜索

```bash
# 测试不同的学习率
for lr in 0.01 0.05 0.1; do
    python main.py finetune_evaluation \
        --config config/SHREC/joint/joint.yaml \
        --base_lr $lr \
        --work_dir work_dir/SHREC/lr_${lr}
done
```

### 熵权重搜索

```bash
# 测试不同的熵权重
for lambda_e in 0.0001 0.001 0.005 0.01; do
    python main.py finetune_evaluation \
        --config config/SHREC/joint/joint.yaml \
        --lambda_entropy $lambda_e \
        --work_dir work_dir/SHREC/entropy_${lambda_e}
done
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 🙏 致谢

本项目受以下优秀工作启发:

- **CTR-GCN**: 基础架构灵感 [[GitHub](https://github.com/Uason-Chen/CTR-GCN)]
- **SHREC'17 Track**: 手势数据集 [[网站](http://www-rech.telecom-lille.fr/shrec2017-hand/)]
- **EgoGesture**: 以自我为中心的手势数据集 [[论文](http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html)]

## 📮 联系方式

如有问题和反馈:
- 在 GitHub 上提交 issue
- 邮箱: [your-email@example.com] (待更新)

---

**注意**: 本 README 反映了项目当前的实验状态。性能数据、模型架构和实现细节可能随开发进展而变化。