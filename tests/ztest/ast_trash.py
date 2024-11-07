import torch
from pykeen.models import TransE
from pykeen.datasets import Nations, FB15k
from pykeen.training import SLCWATrainingLoop, TrainingCallback
from pykeen.optimizers import SGD
from pykeen.losses import MarginRankingLoss
from pykeen.evaluation import RankBasedEvaluator
from pykeen.training.callbacks import EvaluationTrainingCallback

# 加载数据集
dataset = FB15k()
training_triples = dataset.training.mapped_triples
# 创建损失函数
loss = MarginRankingLoss(margin=1, reduction="mean")
# 创建模型
model = TransE(
    triples_factory=dataset.training,
    embedding_dim=50,  # 嵌入维度
    scoring_fct_norm=1,  # 打分函数的范数
    power_norm=False,  # 是否使用幂范数
    entity_initializer="xavier_uniform",  # 实体初始化方法
    relation_initializer="xavier_uniform",  # 关系初始化方法
    entity_constrainer="normalize",  # 实体约束器
    loss=loss

).to("cuda")  # 将模型移动到 CUDA 设备

# 创建优化器
optimizer = SGD(params=model.parameters(), lr=0.01)
# eval_callback = EvaluationTrainingCallback(evaluation_triples=dataset.testing.mapped_triples,additional_filter_triples=dataset.training.mapped_triples)
# 创建训练循环
training_loop = SLCWATrainingLoop(
    triples_factory=dataset.training,
    model=model,
    optimizer=optimizer,

)

# 进行训练
training_loop.train(
    # callbacks=[eval_callback],
    triples_factory=dataset.training,
    num_epochs=100,  # 训练轮数
    batch_size=32,  # 批量大小
)

# 创建评估器
evaluator = RankBasedEvaluator(filtered=True)

# 进行评估
results = evaluator.evaluate(
    model=model,
    mapped_triples=dataset.testing.mapped_triples
)

# 输出评估结果
print(results.to_dict()["both"]["optimistic"])
