import pickle
import numpy as np
from tqdm import tqdm
import os
import sys

print("Starting Three-Stream Ensemble Evaluation (Joint + Bone + J-Motion)...")

# =================================================================
# 1. 配置路径 (已根据您的要求更新)
# =================================================================

# Stream 1: Joint (关节)
joint_path = '/Users/gzp/Desktop/exp/DSA-HGN1/work_dir/SHREC/joint_finetune/test_result.pkl'

# Stream 2: Bone (骨骼)
bone_path = '/Users/gzp/Desktop/exp/DSA-HGN1/work_dir/SHREC/bone/test_result.pkl'

# Stream 3: J-Motion (关节运动)
motion_path = '/Users/gzp/Desktop/exp/DSA-HGN1/work_dir/SHREC/jmotion_finetune/test_result.pkl'

# 标签文件路径
label_path = '/Users/gzp/Desktop/exp/DATA/SHREC2017_data/val_label.pkl'

# =================================================================
# 2. 融合策略
# =================================================================
# 初始权重设置 [Joint, Bone, Motion]
# 建议起点: [1.0, 1.0, 0.8]
alpha = [1.0, 1.0, 0.8]

print(f"Initial Fusion Weights -> Joint: {alpha[0]}, Bone: {alpha[1]}, Motion: {alpha[2]}")


def load_pkl(path):
    if not os.path.exists(path):
        # 尝试查找 best_result.pkl 作为备选
        alt_path = path.replace('test_result.pkl', 'best_result.pkl')
        if os.path.exists(alt_path):
            print(f"[Info] '{path}' not found. Using '{alt_path}' instead.")
            return pickle.load(open(alt_path, 'rb'))
        else:
            raise FileNotFoundError(f"Cannot find result file at {path}")

    with open(path, 'rb') as f:
        return pickle.load(f)


# =================================================================
# 3. 加载数据
# =================================================================
try:
    print("-" * 60)
    # 1. 加载 Joint
    print(f"1. Loading Joint    <- {joint_path}")
    r1_dict = load_pkl(joint_path)

    # 2. 加载 Bone
    print(f"2. Loading Bone     <- {bone_path}")
    r2_dict = load_pkl(bone_path)

    # 3. 加载 Motion
    print(f"3. Loading J-Motion <- {motion_path}")
    r3_dict = load_pkl(motion_path)

    # 4. 加载标签
    print(f"4. Loading labels   <- {label_path}")
    with open(label_path, 'rb') as f:
        label_data = pickle.load(f)
        if isinstance(label_data, tuple) or isinstance(label_data, list):
            sample_names = label_data[0]
            true_labels = label_data[1]
        elif isinstance(label_data, dict):
            sample_names = list(label_data.keys())
            true_labels = list(label_data.values())
        else:
            raise ValueError("Unknown label file format")

    print("-" * 60)

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("请检查路径是否正确，或者对应的 .pkl 文件是否已经生成。")
    sys.exit(1)  # 遇到错误直接退出，避免后续报错

# =================================================================
# 4. 执行融合评估
# =================================================================
right_num = 0
total_num = 0
right_num_5 = 0

print(f"Evaluating Fusion on {len(sample_names)} samples...")

for i in tqdm(range(len(sample_names))):
    name = sample_names[i]
    label = int(true_labels[i])

    # 检查样本完整性
    if name not in r1_dict or name not in r2_dict or name not in r3_dict:
        # print(f"Warning: Sample {name} missing in one of the streams.")
        continue

    r1 = r1_dict[name]  # Joint
    r2 = r2_dict[name]  # Bone
    r3 = r3_dict[name]  # Motion

    # --- 核心融合公式 ---
    r = r1 * alpha[0] + r2 * alpha[1] + r3 * alpha[2]
    # ------------------

    # Top-1
    pred = np.argmax(r)
    if pred == label:
        right_num += 1

    # Top-5
    rank_5 = r.argsort()[-5:]
    if label in rank_5:
        right_num_5 += 1

    total_num += 1

if total_num > 0:
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print('\n' + '=' * 50)
    print(f'Final Fusion Accuracy (Joint + Bone + J-Motion)')
    print(f'Weights: {alpha}')
    print('=' * 50)
    print(f'Top-1 Accuracy: {acc * 100:.2f}%')
    print(f'Top-5 Accuracy: {acc5 * 100:.2f}%')
    print('=' * 50)
else:
    print("Error: No common samples found.")
    sys.exit(1)

# =================================================================
# 5. 自动权重搜索 (Grid Search)
# =================================================================
print("\n" + "=" * 20 + " Starting Grid Search " + "=" * 20)
print("(Fixing Joint=1.0, searching Bone & Motion weights)")

best_acc = 0
best_params = []

# 搜索范围：0.2 到 2.0
weights_range = [i / 10.0 for i in range(2, 21, 2)]

for w_bone in weights_range:
    for w_motion in weights_range:
        current_right = 0

        # 简化循环以加速搜索
        for i in range(len(sample_names)):
            name = sample_names[i]
            if name not in r1_dict: continue

            label = int(true_labels[i])

            # 获取分数
            s1 = r1_dict[name]
            s2 = r2_dict[name]
            s3 = r3_dict[name]

            # 融合
            score = s1 * 1.0 + s2 * w_bone + s3 * w_motion

            if np.argmax(score) == label:
                current_right += 1

        curr_acc = current_right / total_num

        if curr_acc > best_acc:
            best_acc = curr_acc
            best_params = [1.0, w_bone, w_motion]
            print(f" -> New Best! Acc: {best_acc * 100:.2f}% | Weights: {best_params}")

print("-" * 50)
print(f"Best Accuracy Found: {best_acc * 100:.2f}%")
print(f"Optimal Weights -> Joint: 1.0, Bone: {best_params[1]}, Motion: {best_params[2]}")
print("-" * 50)