import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def knn_monitor(net, memory_data_loader, test_data_loader, epoch, k=200, t=0.1, hide_progress=False):
    net.eval()

    # ✅ 获取设备 (支持 CUDA, MPS, CPU)
    device = next(net.parameters()).device

    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []

    with torch.no_grad():
        # Generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting', disable=hide_progress):
            # ✅ 使用动态设备
            data = data.to(device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        target_bank = torch.cat(target_bank, dim=0).contiguous().to(device)

        # ✅ 从实际的 target 中获取类别数
        classes = int(target_bank.max().item()) + 1

        # ✅ 确保 k 不超过样本数量
        num_samples = feature_bank.size(1)
        knn_k = min(k, num_samples)

        if knn_k < k:
            print(f'Warning: k={k} is larger than sample size={num_samples}, using k={knn_k} instead.')

        # Loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
        for data, target in test_bar:
            # ✅ 使用动态设备
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, target_bank, classes, knn_k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Accuracy': total_top1 / total_num * 100})

    return total_top1 / total_num * 100


# knn predict
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # Compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # Counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
    # Weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels