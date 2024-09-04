import os

import numpy as np
import scipy
import torch
from torch import LongTensor
from torch.utils.data import Dataset, DataLoader


class LcwaDataSet(Dataset):
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

        # Convert the set of triples to a NumPy array
        all_triples_np = np.array(list(self.all_triples), dtype=np.int64)  # Use int64 to match torch.long

        # Convert the NumPy array to a LongTensor
        all_triples_tensor = torch.from_numpy(all_triples_np).to(torch.long)
        self.from_triples(all_triples_tensor, num_entities=self.num_entities, num_relations=self.num_relations)

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

    def from_triples(
            self,
            mapped_triples,
            *,
            num_entities: int,
            num_relations: int,
            target: int | None = None,
            **kwargs,
    ):
        if target is None:
            target = 2
        # 如果 target 没有被指定，默认值设置为 2，表示目标是 tail 列（即三元组中的第三列）。
        mapped_triples = mapped_triples.numpy()
        # range(3) 生成 [0, 1, 2]，表示三元组的三列。通过 difference({target}) 将目标列排除，得到非目标列。假设 target=2，那么 other_columns 将为 [0, 1]。
        other_columns = sorted(set(range(3)).difference({target}))

        unique_pairs, pair_idx_to_triple_idx = np.unique(mapped_triples[:, other_columns], return_inverse=True, axis=0)
        num_pairs = unique_pairs.shape[0]
        tails = mapped_triples[:, target]
        target_size = num_relations if target == 1 else num_entities
        compressed = scipy.sparse.coo_matrix(
            (np.ones(mapped_triples.shape[0], dtype=np.float32), (pair_idx_to_triple_idx, tails)),
            shape=(num_pairs, target_size),
        )
        '''
        # 假设我们有以下的 pair_idx_to_triple_idx 和 tails
            pair_idx_to_triple_idx = np.array([0, 0, 1, 2, 3])
            tails = np.array([1, 2, 2, 3, 3])

            [[0. 1. 1. 0.]
             [0. 0. 1. 0.]
             [0. 0. 0. 1.]
             [0. 0. 0. 1.]]
        '''
        # convert to csr for fast row slicing
        compressed = compressed.tocsr()
        '''
            [[0. 1. 1. 0.]
             [0. 0. 1. 0.]
             [0. 0. 0. 1.]
             [0. 0. 0. 1.]]
             CSR 格式使用以下三个属性来存储稀疏矩阵：
                data: 存储所有非零元素的值。
                indices: 存储与非零元素对应的列索引。
                indptr: 指向每行数据起始位置的索引（行指针）。

                data: [1., 1., 1., 1., 1]
                indices: [1, 2, 2, 3, 3]
                indptr: [0, 2, 3, 4, 5]
                indptr[0] = 0: 第 0 行的非零元素从 data[0] 开始，indices[0:2] 表示第 0 行有两个非零元素，列索引为 1 和 2。
        '''
        self.pairs = unique_pairs
        self.compressed = compressed

    def __len__(self) -> int:  # noqa: D105
        return self.pairs.shape[0]

    def __getitem__(self, item: int):  # noqa: D105
        # (array([0, 0]), array([1., 1., 0., 0.]))  # 对应的 (head, relation) 和 tails 向量
        return self.pairs[item], np.asarray(self.compressed[item, :].todense())[0, :]


if __name__ == '__main__':
    data_path = './data'  # 数据集路径
    train_dataset = LcwaDataSet(data_path, mode='train')
    i = 0
    for pair, compressed in train_dataset:
        if i > 30:
            break
        i = i + 1
        indices_of_ones = np.where(compressed == 1)[0]
        print(f"{pair}-{indices_of_ones}")
    for triplet in train_dataset.all_triples:
        if triplet[0] == 0:
            print(triplet)
    print(train_dataset.num_entities)
    train_dataloader = DataLoader(train_dataset, batch_size=28, shuffle=True, num_workers=4)

    for batch in train_dataloader:
        pairs, labels = batch
        print(pairs.shape)
        print(labels.shape)
        break
