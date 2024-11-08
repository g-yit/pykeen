"""
@FileName：replicate_conkb.py
@Description：
@Author：zhangyt\n
@Time：2024/11/8 8:59
"""

import pykeen.datasets
import pykeen.models
import pykeen.training
import pykeen.optimizers
import pykeen.evaluation
from pykeen.losses import SoftplusLoss
from pykeen.regularizers import PowerSumRegularizer
from pykeen.sampling import BernoulliNegativeSampler

# Load the FB15k-237 dataset
dataset = pykeen.datasets.FB15k237()
regularizer = PowerSumRegularizer(
    weight=0.0005,
    p=2.0,
    apply_only_once=True,
    normalize=False
)
# Set up the loss function
loss = SoftplusLoss(reduction='mean')
# Initialize the ConvKB model with the specified parameters
model = pykeen.models.ConvKB(
    embedding_dim=100,
    num_filters=50,
    hidden_dropout_rate=0.0,
    entity_initializer='xavier_uniform',
    relation_initializer='xavier_uniform',
    triples_factory=dataset.training,
    regularizer=regularizer,
    loss=loss,
).to("cuda")

# Set up the regularizer


# Set up the optimizer
optimizer = pykeen.optimizers.Adam(
    params=model.get_grad_params(),
    lr=5e-06
)

# Set up the negative sampler
negative_sampler = BernoulliNegativeSampler(
    mapped_triples=dataset.training.mapped_triples
)

# Configure the training loop
training_loop = pykeen.training.SLCWATrainingLoop(
    model=model,
    triples_factory=dataset.training,
    optimizer=optimizer,
    negative_sampler=negative_sampler,
    negative_sampler_kwargs=dict(num_negs_per_pos=1)
)

# Train the model
training_loop.train(
    num_epochs=200,
    batch_size=256,
    triples_factory=dataset.training,
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
print(f"Mean Rank: {results.get_metric('mean_rank')['optimistic']}")
print(f"Hits@10: {results.get_metric('hits_at_k')['optimistic'][10]}")
print(f"Mean Reciprocal Rank: {results.get_metric('mean_reciprocal_rank')['optimistic']}")
