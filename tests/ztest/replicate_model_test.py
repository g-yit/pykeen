from pykeen.pipeline import pipeline

def replicate_transe():

    # 使用 PyKEEN 的 pipeline 方法进行训练，并使用 CUDA 设备
    result = pipeline(
        dataset="fb15k",  # 数据集名称
        model="TransE",   # 模型名称
        model_kwargs={
            "embedding_dim": 50,             # 嵌入维度
            "scoring_fct_norm": 1,           # 打分函数的范数
            "power_norm": False,             # 是否使用幂范数
            "entity_initializer": "xavier_uniform",  # 实体初始化方法
            "relation_initializer": "xavier_uniform",  # 关系初始化方法
            "entity_constrainer": "normalize"         # 实体约束器
        },
        optimizer="SGD",  # 优化器名称
        optimizer_kwargs={
            "lr": 0.01  # 学习率
        },
        loss="MarginRankingLoss",  # 损失函数
        loss_kwargs={
            "reduction": "mean",  # 损失函数的缩减方法
            "margin": 1           # 损失函数的边距
        },
        training_loop="SLCWA",  # 训练循环方法
        negative_sampler="basic",  # 负采样器名称
        negative_sampler_kwargs={
            "num_negs_per_pos": 1  # 每个正例的负例数量
        },
        training_kwargs={
            "num_epochs": 1000,  # 训练轮数
            "batch_size": 32     # 批量大小
        },
        evaluator_kwargs={
            "filtered": True  # 使用过滤评估
        },
        device='cuda'  # 设置为 'cuda' 使用 GPU 进行训练
    )

    # 输出评估结果
    print(result.metric_results.to_dict())
