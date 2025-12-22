import pickle
import numpy as np
import os


joint_path = './work_dir/SHREC/hyperhand_joint/best_result.pkl'  # 假设脚本自动找到了 best
bone_path = './work_dir/SHREC/hyperhand_bone/best_result.pkl'  # <--- 请检查这里能否换成 best_result.pkl
motion_path = './work_dir/SHREC/hyperhand_motion/best_result.pkl'
label_path = '/DATA/SHREC2017_data/val_label.pkl'

# 使用刚刚搜索到的最佳权重
# 如果你找到了 Bone 的 best_result，建议重新把这个改回 [1.0, 1.0, 0.8] 跑一下搜索
best_alpha = [1.0, 0.2, 0.2]

# SHREC 14 手势或者是 28 手势？
# 这里为了显示方便，如果不知道具体类别名，直接显示 ID
# 如果你有类别列表，可以填在这里，例如:
# class_names = ["Grab", "Tap", "Expand", "Pinch", ...]
class_names = None

# -----------------------------------------------------------

def load_data(path):
    # 简单的加载逻辑，自动找 best
    if not os.path.exists(path):
        alt = path.replace('test_result.pkl', 'best_result.pkl')
        if os.path.exists(alt):
            return pickle.load(open(alt, 'rb')), alt
    return pickle.load(open(path, 'rb')), path


try:
    print(f"Loading files...")
    r1, p1 = load_data(joint_path)
    r2, p2 = load_data(bone_path)
    r3, p3 = load_data(motion_path)
    print(f"Loaded: \nJoint: {p1}\nBone:  {p2}\nMotion: {p3}")

    with open(label_path, 'rb') as f:
        label_data = pickle.load(f)
        sample_names = label_data[0]
        true_labels = label_data[1]

    total = 0
    errors = []

    # 混淆矩阵计数 (True Label -> Pred Label)
    confusion_list = {}

    for i in range(len(sample_names)):
        name = sample_names[i]
        label = int(true_labels[i])

        if name not in r1 or name not in r2 or name not in r3:
            continue

        # 融合
        score = r1[name] * best_alpha[0] + \
                r2[name] * best_alpha[1] + \
                r3[name] * best_alpha[2]

        pred = np.argmax(score)

        if pred != label:
            # 记录错误
            errors.append((name, label, pred))

            pair = (label, pred)
            confusion_list[pair] = confusion_list.get(pair, 0) + 1

        total += 1

    acc = 1 - (len(errors) / total)
    print(f"\nTotal Samples: {total}")
    print(f"Accuracy with weights {best_alpha}: {acc * 100:.2f}%")
    print(f"Total Errors: {len(errors)}")

    print("\n" + "=" * 20 + " 错误分布 (Top 5 混淆对) " + "=" * 20)
    print("格式: (真实标签) -> (预测错误为) : 错误次数")

    # 按错误次数排序
    sorted_conf = sorted(confusion_list.items(), key=lambda x: x[1], reverse=True)

    for i, (pair, count) in enumerate(sorted_conf[:10]):
        true_lbl, pred_lbl = pair
        t_name = f"Class {true_lbl}"
        p_name = f"Class {pred_lbl}"
        print(f"{i + 1}. {t_name} -> {p_name} : {count} 次")

    print("\n" + "=" * 40)
    print("分析建议:")
    if len(errors) > 0:
        top_error = sorted_conf[0]
        print(f"最大的痛点是: 类别 {top_error[0][0]} 经常被误判为 类别 {top_error[0][1]}")
        print("如果这两个动作很像（比如向左/向右，或者抓取/捏合），说明空间特征提取还需要加强（Bone流）。")

except Exception as e:
    print(e)