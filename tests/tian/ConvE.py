import numpy as np
import torch
from torch.nn.init import xavier_normal_

from torch.nn import functional as F, Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

from tests.tian.LcwaDataSet import LcwaDataSet


class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, 200, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, 200, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = 20
        self.emb_dim2 = 200 // self.emb_dim1
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(200)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(9728, 200)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred


def train(model, dataloader, optimizer, device, epochs=100):
    """
    训练TransE模型
    :param model: TransE模型
    :param dataloader: 数据加载器
    :param optimizer: 优化器
    :param device: 设备
    :param epochs: 训练轮数
    """
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            optimizer.zero_grad()
            pair, label = batch
            pair = pair.to(device)
            label = label.to(device)
            pair = pair.transpose(0, 1)
            pred = model.forward(pair[0], pair[1])
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")


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
            # print(max(pair[0]))
            # print(max(pair[1]))
            pair = pair.transpose(0, 1)
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


def main():
    # 配置参数
    data_path = './FB15K237'  # 数据集路径
    embedding_dim = 100
    margin = 1.0
    norm = 1
    batch_size = 128
    epochs = 100
    learning_rate = 0.001
    device = torch.device('cuda')

    # 准备训练数据
    train_dataset = LcwaDataSet(data_path, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 初始化模型
    model = ConvE(num_entities=train_dataset.num_entities,
                  num_relations=train_dataset.num_relations,
                  ).to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    train(model, train_dataloader, optimizer, device, epochs=epochs)

    # 准备验证数据
    valid_dataset = LcwaDataSet(data_path, mode='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # 评估模型
    print("Evaluating on validation set:")
    evaluate(model, valid_dataloader, device)
    # 保存模型
    torch.save(model.state_dict(), 'conve_fb237k.th')
    print("模型已保存为 conve.pth")


if __name__ == '__main__':
    valid_dataset = LcwaDataSet(data_path = './FB15K237', mode='test')
    valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4)
    model = ConvE(num_entities=valid_dataset.num_entities, num_relations=valid_dataset.num_relations)
    device = torch.device('cuda')
    model.load_state_dict(torch.load('conve_fb237k.th'))
    model=model.to(device)
    model.eval()
    evaluate(model, valid_dataloader, device=torch.device('cuda'))

    # main()
    # # 准备验证数据
    # valid_dataset = LcwaDataSet('./FB15K237', mode='valid')
    # print(valid_dataset.num_entities)
    # print(valid_dataset.num_relations)
    # # 假设 self.triples 已经是一个列表，形如 [(h1, r1, t1), (h2, r2, t2), ...]
    #
    # # 提取所有关系索引 r 的列表
    # relations = [triple[1] for triple in valid_dataset.triples]
    #
    # # 计算关系索引的最大值
    # max_relation = max(relations)
    # print(max_relation)
