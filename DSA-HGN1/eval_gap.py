import sys
import os
from torchlight import import_class


def run_gap_evaluation():
    # 1. 指定要使用的 Processor
    # 对应命令行的 'finetune_evaluation'
    Processor = import_class('processor.finetune_evaluation.FT_Processor')

    # 2. 定义临时参数列表 (覆盖 yaml 配置)
    # 我们不仅读取原配置，还通过命令行参数强制覆盖关键设置
    simulated_argv = [
        # --- 基础配置 ---
        '--config', 'config/SHREC/bone.yaml/bone_fi.yaml',

        # --- 核心策略：回滚与重演 ---
        # 加载 Epoch 25 的最佳权重 (请确认路径正确)
        '--weights', './work_dir/SHREC/bone.yaml/epoch025_acc87.14_model.pt',
        # 从 25 开始，跳过 Warm-up
        '--start_epoch', '25',
        # 只跑到 40 (我们的目标是检查 26-40 之间的峰值)
        '--num_epoch', '40',

        # --- 核心目标：密集评估 ---
        '--eval_interval', '1',  # 每一轮都评估！
        '--save_interval', '1',  # 每一轮都保存权重
        '--log_interval', '10',  # 频繁打印日志以便观察

        # --- 环境隔离 ---
        # 使用一个新的临时目录，避免覆盖主实验的 epoch40 权重
        '--work_dir', './work_dir/SHREC/gap_check_epoch26_to_40',

        # --- 保持原有训练参数 ---
        '--phase', 'train',
        '--optimizer', 'SGD',
        '--base_lr', '0.05',
        '--step', '30',  # 确保在 Ep 30 发生衰减
        '--lambda_entropy', '0.001',
        '--lambda_ortho', '0.1'
    ]

    print("----------------------------------------------------------------")
    print(f"[Gap Eval] 开始回滚至 Epoch 25 并密集扫描至 Epoch 40...")
    print(f"[Gap Eval] 结果将保存在: ./work_dir/SHREC/gap_check_epoch26_to_40")
    print("----------------------------------------------------------------")

    # 3. 初始化并启动处理器
    p = Processor(simulated_argv)
    p.start()


if __name__ == '__main__':
    run_gap_evaluation()