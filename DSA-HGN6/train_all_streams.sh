#!/bin/bash

# ============================================================
# DSA-HGN 四流训练脚本
# SHREC2017 手势识别数据集
# ============================================================

echo "=========================================="
echo "DSA-HGN Multi-Stream Training Pipeline"
echo "=========================================="
echo ""

# 设置Python环境（根据实际情况修改）
# source activate your_env_name

# GPU设置
export CUDA_VISIBLE_DEVICES=0

# ============================================================
# 阶段 1: 训练 Joint 流（基准模型）
# ============================================================
echo "[1/4] Training Joint Stream..."
python main.py finetune_evaluation \
  --config config/SHREC/joint/joint.yaml \
  --phase train \
  --device 0

echo "Joint stream training completed!"
echo ""

# ============================================================
# 阶段 2: 训练 Bone 流（可选知识蒸馏）
# ============================================================
echo "[2/4] Training Bone Stream..."
python main.py finetune_evaluation \
  --config config/SHREC/bone/bone.yaml \
  --phase train \
  --device 0

# 如果要使用知识蒸馏（用Joint流指导Bone流）：
# python main.py finetune_evaluation \
#   --config config/SHREC/bone/bone.yaml \
#   --phase train \
#   --device 0 \
#   --teacher_weights ./work_dir/SHREC/joint_v2/best_model.pt \
#   --lambda_kd 0.3

echo "Bone stream training completed!"
echo ""

# ============================================================
# 阶段 3: 训练 Joint Motion 流
# ============================================================
echo "[3/4] Training Joint Motion Stream..."
python main.py finetune_evaluation \
  --config config/SHREC/joint_motion/joint_motion.yaml \
  --phase train \
  --device 0

echo "Joint Motion stream training completed!"
echo ""

# ============================================================
# 阶段 4: 训练 Bone Motion 流
# ============================================================
echo "[4/4] Training Bone Motion Stream..."
python main.py finetune_evaluation \
  --config config/SHREC/bone_motion/bone_motion.yaml \
  --phase train \
  --device 0

echo "Bone Motion stream training completed!"
echo ""

# ============================================================
# 总结结果
# ============================================================
echo "=========================================="
echo "Training Summary"
echo "=========================================="

echo ""
echo "Extracting best accuracies from log files..."
echo ""

# Joint流
if [ -f "./work_dir/SHREC/joint_v2/log.txt" ]; then
    JOINT_ACC=$(grep "Best Top1:" ./work_dir/SHREC/joint_v2/log.txt | tail -1 | awk '{print $3}')
    echo "Joint Stream:        ${JOINT_ACC}"
fi

# Bone流
if [ -f "./work_dir/SHREC/bone_v2/log.txt" ]; then
    BONE_ACC=$(grep "Best Top1:" ./work_dir/SHREC/bone_v2/log.txt | tail -1 | awk '{print $3}')
    echo "Bone Stream:         ${BONE_ACC}"
fi

# Joint Motion流
if [ -f "./work_dir/SHREC/joint_motion_v2/log.txt" ]; then
    JM_ACC=$(grep "Best Top1:" ./work_dir/SHREC/joint_motion_v2/log.txt | tail -1 | awk '{print $3}')
    echo "Joint Motion Stream: ${JM_ACC}"
fi

# Bone Motion流
if [ -f "./work_dir/SHREC/bone_motion_v2/log.txt" ]; then
    BM_ACC=$(grep "Best Top1:" ./work_dir/SHREC/bone_motion_v2/log.txt | tail -1 | awk '{print $3}')
    echo "Bone Motion Stream:  ${BM_ACC}"
fi

echo ""
echo "All models saved in ./work_dir/SHREC/"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Review individual stream performance above"
echo "2. Adjust ensemble weights in multistream_ensemble.yaml"
echo "3. Run ensemble testing:"
echo "   python main.py multistream_test --config config/SHREC/multistream_ensemble.yaml"
echo ""
echo "Expected ensemble performance: ~97-98% (based on typical multi-stream fusion gains)"
echo "=========================================="
echo ""

# ============================================================
# 可选：自动运行集成测试
# ============================================================
read -p "Run ensemble testing now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running multi-stream ensemble testing..."
    python main.py multistream_test \
      --config config/SHREC/multistream_ensemble.yaml \
      --phase test
fi

echo "Done!"