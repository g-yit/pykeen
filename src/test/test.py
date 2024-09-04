import torch

from src.pykeen.pipeline import pipeline

if __name__ == '__main__':
    # Step 1: Get triples
    from src.pykeen.datasets import FB15k

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FB15k(create_inverse_triples=True)
    # Step 2: Configure the model
    from src.pykeen.models import ConvE

    triples_factory = dataset.training
    model = ConvE(
        triples_factory=triples_factory,
        embedding_dim=200,
        input_channels=1,
        output_channels=32,
        embedding_height=10,
        embedding_width=20,
        kernel_height=3,
        kernel_width=3,
        input_dropout=0.2,
        feature_map_dropout=0.2,
        output_dropout=0.3,
    ).to(device)
    # Step 3: Configure the loop
    from torch.optim import Adam

    optimizer = Adam(params=model.get_grad_params())
    from src.pykeen.training import LCWATrainingLoop

    training_loop = LCWATrainingLoop(model=model, optimizer=optimizer, triples_factory=dataset.training,)
    # Step 4: Train
    losses = training_loop.train(num_epochs=50, batch_size=256, triples_factory=dataset.training)
    # Step 5: Evaluate the model
    from src.pykeen.evaluation import RankBasedEvaluator

    evaluator = RankBasedEvaluator()
    metric_result = evaluator.evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=dataset.training.mapped_triples,
        batch_size=8192,
    )

    print(metric_result.to_dict())
