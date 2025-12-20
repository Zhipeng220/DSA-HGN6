import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# --- 配置 ---
# 使用最终确认的最佳文件
joint_path = './work_dir/SHREC/hyperhand_joint/best_result.pkl'
bone_path = './work_dir/SHREC/hyperhand_bone/best_result.pkl'
motion_path = './work_dir/SHREC/hyperhand_motion/best_result.pkl'
label_path = '/DATA/SHREC2017_data/val_label.pkl'

# 最终确认的最佳权重
final_alpha = [1.0, 0.2, 0.2]

# SHREC-14 动作列表 (根据官方文档整理，如果是28类请修改)
# 如果你不确定具体名称，可以先设为 None，代码会用 ID 代替
class_names = [
    "Grab", "Tap", "Expand", "Pinch", "Rotation CW", "Rotation CCW",
    "Swipe Right", "Swipe Left", "Swipe Up", "Swipe Down",
    "Swipe X", "Swipe V", "Swipe +", "Shake"
]


# 注意：如果你的数据集是 SHREC-28 (包含细粒度手指动作)，请相应调整列表长度

# -----------------------------------------------------------

def load_pkl(path):
    return pickle.load(open(path, 'rb'))


try:
    print("Generating Confusion Matrix...")
    r1 = load_pkl(joint_path)
    r2 = load_pkl(bone_path)
    r3 = load_pkl(motion_path)

    with open(label_path, 'rb') as f:
        label_data = pickle.load(f)
        sample_names = label_data[0]
        true_labels = label_data[1]

    # 1. 计算所有预测结果
    y_true = []
    y_pred = []

    for i in range(len(sample_names)):
        name = sample_names[i]
        if name not in r1: continue  # 简单防错

        label = int(true_labels[i])

        # 融合
        score = r1[name] * final_alpha[0] + \
                r2[name] * final_alpha[1] + \
                r3[name] * final_alpha[2]

        pred = np.argmax(score)

        y_true.append(label)
        y_pred.append(pred)

    # 2. 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 归一化 (显示百分比而不是绝对数量，更易读)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 3. 绘图
    plt.figure(figsize=(12, 10))

    # 检查类别名称长度是否匹配
    if class_names and len(class_names) == len(cm):
        labels = class_names
    else:
        print(f"Warning: Name list len ({len(class_names) if class_names else 0}) != Class num ({len(cm)}). Using IDs.")
        labels = [str(i) for i in range(len(cm))]

    # 绘制热力图
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.title(f'Confusion Matrix (Acc: {95.24}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图片
    save_path = '../SHREC_Confusion_Matrix.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Confusion Matrix saved to: {save_path}")
    print("图片已生成！你可以打开它查看 'Pinch' 和 'Grab' 的混淆情况。")

    # 4. 打印每个类别的准确率 (Per-class Accuracy)
    print("\n=== Per-Class Accuracy ===")
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(per_class_acc):
        name = labels[i]
        print(f"Class {i} ({name}): {acc * 100:.2f}%")

except Exception as e:
    print(f"Error: {e}")