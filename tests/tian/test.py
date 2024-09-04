import torch

# pred = torch.tensor([[1, 2, 3]
#                         , [4, 5, 6]])
# true_label = torch.tensor([[1, 0, 1],
#                            [0, 1, 0]])
# pred_rank = torch.argsort(pred, dim=1, descending=True)
#
# for i in range(pred.size(0)):  # 遍历每个样本
#     true_indices = true_label[i].nonzero(as_tuple=True)[0]  # 获取正确标签的索引
#     for true_idx in true_indices:
#         rank = (pred_rank[i] == true_idx).nonzero(as_tuple=True)[0].item() + 1
#         print(f"Sample {i}, True label {true_idx}, Rank: {rank}")


import torch
from tqdm import tqdm

# 模拟预测值和真实标签
pred = torch.tensor([[1, 2, 3],
                     [4, 5, 6]])

true_label = torch.tensor([[1, 0, 1],
                           [0, 1, 0]])


#
#
#
# hits_at_1 = 0
# hits_at_3 = 0
# hits_at_10 = 0
#
# # 计算排序
# pred_rank = torch.argsort(pred, dim=1, descending=True)
# # 遍历每个样本
# min_rank = 0
# for i in range(pred.size(0)):
#     # 获取正确标签的索引
#     true_indices = true_label[i].nonzero(as_tuple=True)[0]
#
#     # 初始化最小排名为一个大数
#     min_rank = float('inf')
#
#     # 遍历所有正确标签，找到最小的排名
#     for true_idx in true_indices:
#         rank = (pred_rank[i] == true_idx).nonzero(as_tuple=True)[0].item() + 1
#         if rank < min_rank:
#             min_rank = rank
#     print(min_rank)
#
#     # 计算 Hits@k
#     if min_rank <= 1:
#         hits_at_1 += 1
#     if min_rank <= 3:
#         hits_at_3 += 1
#     if min_rank <= 10:
#         hits_at_10 += 1
#
# # 计算命中率
# total_samples = pred.size(0)
# hits_at_1_ratio = hits_at_1 / total_samples
# hits_at_3_ratio = hits_at_3 / total_samples
# hits_at_10_ratio = hits_at_10 / total_samples
#
# print(f"Hits@1: {hits_at_1_ratio:.4f}")
# print(f"Hits@3: {hits_at_3_ratio:.4f}")
# print(f"Hits@10: {hits_at_10_ratio:.4f}")


def evaluate(model, dataloader, device, k_values=[1, 3, 10]):
    """
    评估TransE模型
    :param model: TransE模型
    :param dataloader: 验证或测试数据加载器
    :param device: 设备
    :param k_values: Hits@K 的 K 值列表
    :return: Mean Rank 和 Hits@K 结果
    """
    model.eval()
    with torch.no_grad():
        ranks = []
        hits_at_k = {k: 0 for k in k_values}

        for batch in tqdm(dataloader, desc="Evaluating"):
            pair, true_label = batch
            # pair 是batch_size*2
            # true_label是batch_size*entity_embedding_num
            pair = pair.to(device)
            true_label = true_label.to(device)

            # 模型预测
            pred = model(pair[0], pair[1])
            # pred batch_size*entity_embedding_num

            # 获取预测值的排序
            pred_rank = torch.argsort(pred, dim=1, descending=True)

            for i in range(pred.size(0)):
                true_indices = true_label[i].nonzero(as_tuple=True)[0]

                # 初始化最小排名为一个大数
                min_rank = float('inf')

                # 遍历所有正确标签，找到最小的排名
                for true_idx in true_indices:
                    rank = (pred_rank[i] == true_idx).nonzero(as_tuple=True)[0].item() + 1
                    if rank < min_rank:
                        min_rank = rank
                ranks.append(min_rank)

                # 计算Hits@K
                for k in k_values:
                    if rank <= k:
                        hits_at_k[k] += 1

        # 计算Mean Rank和Hits@K
        mean_rank = sum(ranks) / len(ranks)
        hits_at_k = {k: v / len(ranks) for k, v in hits_at_k.items()}

        print(f"Mean Rank: {mean_rank:.4f}")
        for k in k_values:
            print(f"Hits@{k}: {hits_at_k[k]:.4f}")

        return mean_rank, hits_at_k
