import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class FB15KDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        """
        初始化数据集，解析实体和关系，并准备三元组。
        :param data_path: 数据集所在目录
        :param mode: 'train', 'valid', 'test'
        """
        self.mode = mode
        self.data = self.load_data(os.path.join(data_path, f"freebase_mtr100_mte100-{mode}.txt"))
        self.entities, self.relations = self.get_entities_relations(data_path)
        self.entity2id = {entity: idx for idx, entity in enumerate(self.entities)}
        self.relation2id = {relation: idx for idx, relation in enumerate(self.relations)}
        self.triples = [(self.entity2id[h], self.relation2id[r], self.entity2id[t]) for h, r, t in self.data]
        self.all_triples = set(self.triples)
        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)

    def load_data(self, file_path):
        """
        加载数据文件
        :param file_path: 文件路径
        :return: 三元组列表
        """
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                data.append((h, r, t))
        return data

    def get_entities_relations(self, data_path):
        """
        获取所有实体和关系
        :param data_path: 数据集目录
        :return: 实体列表，关系列表
        """
        entities = set()
        relations = set()
        for mode in ['train', 'valid', 'test']:
            file_path = os.path.join(data_path, f"freebase_mtr100_mte100-{mode}.txt")
            with open(file_path, 'r') as f:
                for line in f:
                    h, r, t = line.strip().split('\t')
                    entities.update([h, t])
                    relations.add(r)
        return sorted(entities), sorted(relations)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        """
        获取一个三元组及其负样本
        :param idx: 索引
        :return: (h, r, t), (h_neg, r, t_neg)
        """
        h, r, t = self.triples[idx]
        # 负采样：随机替换头或尾实体
        corrupt_head = random.choice([True, False])
        if corrupt_head:
            h_neg = random.randint(0, self.num_entities - 1)
            while (h_neg, r, t) in self.all_triples:
                h_neg = random.randint(0, self.num_entities - 1)
            return torch.tensor([h, r, t], dtype=torch.long), torch.tensor([h_neg, r, t], dtype=torch.long)
        else:
            t_neg = random.randint(0, self.num_entities - 1)
            while (h, r, t_neg) in self.all_triples:
                t_neg = random.randint(0, self.num_entities - 1)
            return torch.tensor([h, r, t], dtype=torch.long), torch.tensor([h, r, t_neg], dtype=torch.long)

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0, norm=1):
        """
        TransE模型定义
        :param num_entities: 实体数量
        :param num_relations: 关系数量
        :param embedding_dim: 嵌入维度
        :param margin: 损失函数的margin
        :param norm: 使用的范数，1或2
        """
        super(TransE, self).__init__()
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm
        # 实体和关系的嵌入
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.init_weights()

    def init_weights(self):
        """
        初始化嵌入权重
        """
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, positive_triples, negative_triples):
        """
        前向传播，计算损失
        :param positive_triples: 正样本三元组(batch_size, 3)
        :param negative_triples: 负样本三元组(batch_size, 3)
        :return: 损失
        """
        pos_h = self.entity_embeddings(positive_triples[:, 0])
        pos_r = self.relation_embeddings(positive_triples[:, 1])
        pos_t = self.entity_embeddings(positive_triples[:, 2])

        neg_h = self.entity_embeddings(negative_triples[:, 0])
        neg_r = self.relation_embeddings(negative_triples[:, 1])
        neg_t = self.entity_embeddings(negative_triples[:, 2])

        # 计算距离
        pos_distance = torch.norm(pos_h + pos_r - pos_t, p=self.norm, dim=1)
        neg_distance = torch.norm(neg_h + neg_r - neg_t, p=self.norm, dim=1)

        # 损失函数
        loss = F.relu(self.margin + pos_distance - neg_distance)
        return torch.mean(loss)

    def predict(self, triples):
        """
        预测三元组的分数
        :param triples: 三元组(batch_size, 3)
        :return: 分数(batch_size)
        """
        h = self.entity_embeddings(triples[:, 0])
        r = self.relation_embeddings(triples[:, 1])
        t = self.entity_embeddings(triples[:, 2])
        distance = torch.norm(h + r - t, p=self.norm, dim=1)
        return distance

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
            positive_triples, negative_triples = batch
            positive_triples = positive_triples.to(device)
            negative_triples = negative_triples.to(device)

            optimizer.zero_grad()
            loss = model(positive_triples, negative_triples)
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
            positive_triples, _ = batch
            positive_triples = positive_triples.to(device)

            for i in range(positive_triples.size(0)):
                head, relation, tail = positive_triples[i]

                # 计算所有可能的尾实体的距离
                all_tails = torch.arange(model.entity_embeddings.num_embeddings).to(device)
                tail_distances = model.predict(torch.stack((head.repeat(all_tails.size(0)), relation.repeat(all_tails.size(0)), all_tails), dim=1))

                # 获取正确的尾实体的排名
                rank = torch.argsort(tail_distances).tolist().index(tail.item()) + 1
                ranks.append(rank)

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
    data_path = './data'  # 数据集路径
    embedding_dim = 100
    margin = 1.0
    norm = 1
    batch_size = 1024
    epochs = 10
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备训练数据
    train_dataset = FB15KDataset(data_path, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 初始化模型
    model = TransE(num_entities=train_dataset.num_entities,
                  num_relations=train_dataset.num_relations,
                  embedding_dim=embedding_dim,
                  margin=margin,
                  norm=norm).to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    train(model, train_dataloader, optimizer, device, epochs=epochs)

    # 准备验证数据
    valid_dataset = FB15KDataset(data_path, mode='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # 评估模型
    print("Evaluating on validation set:")
    evaluate(model, valid_dataloader, device)
    # 保存模型
    torch.save(model.state_dict(), 'transE_fb15k.pth')
    print("模型已保存为 transE_fb15k.pth")

if __name__ == '__main__':
    main()

