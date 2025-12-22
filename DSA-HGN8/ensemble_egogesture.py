import pickle
import numpy as np
from tqdm import tqdm

print("Starting Two-Stream Ensemble Evaluation (Joint + Bone)...")

# --- 1. 配置路径 (根据您的设置) ---

# 🔴 1a. 指向您的“关节点”微调 (finetune) 结果目录
joint_path = 'work_dir/egogesture/aimclr_finetune_joint/'

# 🔴 1b. 指向您的“骨骼”微调 (finetune) 结果目录
bone_path = 'work_dir/egogesture/aimclr_finetune_bone/'

# 🔴 1c. 指向您的验证集标签 (val_label.pkl)
label_path = '/Users/gzp/Desktop/exp/CTR-GCN-main/data/egogesture/val_label.pkl'

# 🔴 1d. 设置融合权重 [关节点, 骨骼]
# (0.5, 0.5) 是最标准的起始点。
alpha = [0.5, 0.5]

# ------------------------------------

print(f"Loading Joint results from: {joint_path}test_result.pkl")
with open(joint_path + 'test_result.pkl', 'rb') as r1:
    r1_dict = pickle.load(r1)

print(f"Loading Bone results from: {bone_path}test_result.pkl")
with open(bone_path + 'test_result.pkl', 'rb') as r2:
    r2_dict = pickle.load(r2)

print(f"Loading labels from: {label_path}")
with open(label_path, 'rb') as f:
    label_data = pickle.load(f)

# 假设 val_label.pkl 是一个包含 [sample_names, label_ids] 的列表
try:
    sample_names = label_data[0]
    true_labels = label_data[1]
    print(f"Loaded {len(true_labels)} labels.")
except Exception as e:
    print(f"Error loading label file: {e}")
    print("Exiting. Please check the structure of your val_label.pkl file.")
    exit()

right_num = total_num = right_num_5 = 0

# 遍历所有样本
for i in tqdm(range(len(sample_names))):
    sample_name = sample_names[i]
    l = true_labels[i]

    # 检查两个模型是否都有这个样本的预测
    if sample_name not in r1_dict or sample_name not in r2_dict:
        print(f"Warning: Sample {sample_name} not found in one of the result files. Skipping.")
        continue

    # --- 关键的融合步骤 ---
    # r11 是“关节点”模型的预测分数
    # r22 是“骨骼”模型的预测分数
    r11 = r1_dict[sample_name]
    r22 = r2_dict[sample_name]

    # 将分数按权重相加
    r = (r11 * alpha[0]) + (r22 * alpha[1])
    # -----------------------

    # 计算 Top-5 准确率
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)

    # 计算 Top-1 准确率
    r = np.argmax(r)
    right_num += int(r == int(l))

    total_num += 1

# 计算最终结果
acc = right_num / total_num
acc5 = right_num_5 / total_num

print('-' * 40)
print('Double-Stream Ensemble Result (Joint + Bone)')
print(f'Weighting: Joint={alpha[0]}, Bone={alpha[1]}')
print(f'Total samples evaluated: {total_num}')
print('-' * 40)
print(f'Top-1 Accuracy: {acc * 100:.2f}%')
print(f'Top-5 Accuracy: {acc5 * 100:.2f}%')
print('-' * 40)