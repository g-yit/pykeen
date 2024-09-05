from src.pykeen.datasets import FB15k
from src.pykeen.models import ConvE
from src.pykeen.triples import TriplesFactory
from src.pykeen.pipeline import pipeline

# 加载FB15K数据集
dataset = FB15k()

# 打印数据集信息
print(
    f"训练集: {dataset.training.num_triples}, 验证集: {dataset.validation.num_triples}, 测试集: {dataset.testing.num_triples}")
# 运行训练管道
result = pipeline(
    model='ConvE',
    dataset='FB15k',
    model_kwargs={
        'embedding_dim': 200,
        'input_channels': 1,
        'output_channels': 32,
        'embedding_height': 10,
        'embedding_width': 20,
        'kernel_height': 3,
        'kernel_width': 3,
        'input_dropout': 0.2,
        'feature_map_dropout': 0.2,
        'output_dropout': 0.3
    },
    optimizer='Adam',
    optimizer_kwargs={
        'lr': 0.001,  # 学习率
    },
    training_loop='LCWA',  # 使用的训练方式，通常为'LCWA'（local closed-world assumption）
    training_kwargs={
        'batch_size': 128,  # 批次大小
        'num_epochs': 500,  # 训练轮数
    },
    stopper='early',
    stopper_kwargs={
        'frequency': 10,
        'patience': 10,
        'metric': 'hits@10',  # 用于早停的评价指标
    },
)

# 训练结果
print(result)
