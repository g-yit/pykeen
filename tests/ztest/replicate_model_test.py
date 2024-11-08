from pykeen.pipeline import pipeline
from tests.ztest.callback.training_callback import ZTrainingCallback


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
def replicate_conve():
    import pykeen.datasets
    import pykeen.models
    import pykeen.training
    import pykeen.optimizers
    import pykeen.evaluation
    from pykeen.losses import BCEAfterSigmoidLoss

    # Load the FB15K dataset
    dataset = pykeen.datasets.FB15k(create_inverse_triples=True)
    # Set up the loss function
    loss = BCEAfterSigmoidLoss(reduction='mean')
    # Initialize the ConvE model with the specified parameters
    model = pykeen.models.ConvE(
        embedding_dim=200,
        # input_channels=1,
        output_channels=32,
        embedding_height=10,
        embedding_width=20,
        kernel_height=3,
        kernel_width=3,
        input_dropout=0.2,
        feature_map_dropout=0.2,
        output_dropout=0.3,
        apply_batch_normalization=True,
        entity_initializer='xavier_normal',
        relation_initializer='xavier_normal',
        triples_factory=dataset.training,
        loss=loss
    ).to("cuda")

    # Set up the optimizer
    optimizer = pykeen.optimizers.Adam(
        params=model.get_grad_params(),
        lr=0.001
    )
    # Configure the training loop
    training_loop = pykeen.training.LCWATrainingLoop(
        model=model,
        triples_factory=dataset.training,
        optimizer=optimizer,
    )
    eval_callback = ZTrainingCallback(evaluation_triples=dataset.validation.mapped_triples,
                                      full_test_evaluation_triples=dataset.testing.mapped_triples,
                                      additional_filter_triples=dataset.training.mapped_triples)

    # Train the model
    training_loop.train(
        triples_factory=dataset.training,
        num_epochs=1000,
        batch_size=128,
        label_smoothing=0.1,
        use_tqdm_batch=False,
        callbacks=[eval_callback],
    )

    # Evaluate the model
    evaluator = pykeen.evaluation.RankBasedEvaluator(filtered=True)
    results = evaluator.evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples
        ]
    )

    # Output the results
    print(results.to_dict())

if __name__ == '__main__':
    replicate_conve()
