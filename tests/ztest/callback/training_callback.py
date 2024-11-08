"""
@FileName：training_callback.py
@Description：
@Author：zhangyt\n
@Time：2024/11/7 22:38
"""
from typing import Any

from class_resolver import HintOrType, OptionalKwargs

from pykeen.evaluation import Evaluator, evaluator_resolver
from pykeen.training import TrainingCallback
from pykeen.training.callbacks import EvaluationTrainingCallback
from pykeen.typing import MappedTriples


class ZTrainingCallback(TrainingCallback):
    def __init__(
            self,
            *,
            evaluation_triples: MappedTriples,
            full_test_evaluation_triples: MappedTriples,
            additional_filter_triples,
            frequency: int = 5,
            full_test_frequency: int = 20,
            evaluator: HintOrType[Evaluator] = None,
            evaluator_kwargs: OptionalKwargs = None,
            prefix: str | None = None,
            **kwargs,
    ):
        """
        Initialize the callback.

        :param evaluation_triples:
            the triples on which to evaluate
        :param frequency:
            the evaluation frequency in epochs
        :param evaluator:
            the evaluator to use for evaluation, cf. `evaluator_resolver`
        :param evaluator_kwargs:
            additional keyword-based parameters for the evaluator
        :param prefix:
            the prefix to use for logging the metrics
        :param kwargs:
            additional keyword-based parameters passed to `evaluate`
        """
        super().__init__()
        self.frequency = frequency
        self.full_test_frequency = full_test_frequency
        self.additional_filter_triples = additional_filter_triples
        self.evaluation_triples = evaluation_triples
        self.evaluator = evaluator_resolver.make(evaluator, evaluator_kwargs)
        self.prefix = prefix
        self.kwargs = kwargs
        self.batch_size = self.kwargs.pop("batch_size", None)
        self.full_test_evaluation_triples = full_test_evaluation_triples
    # docstr-coverage: inherited
    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        if epoch >= 1 and epoch % self.frequency == 0:
            print("")
            print("validation ........")
            result = self.evaluator.evaluate(
                additional_filter_triples=self.additional_filter_triples,
                model=self.model,
                use_tqdm=False,
                mapped_triples=self.evaluation_triples,
                device=self.training_loop.device,
                batch_size=self.evaluator.batch_size or self.batch_size,
                **self.kwargs,
            )
            mmr = result.to_dict()["both"]["optimistic"]["inverse_harmonic_mean_rank"]
            mr = result.to_dict()["both"]["optimistic"]["arithmetic_mean_rank"]
            hits1 = result.to_dict()["both"]["optimistic"]["hits_at_1"]
            hits5 = result.to_dict()["both"]["optimistic"]["hits_at_5"]
            hits10 = result.to_dict()["both"]["optimistic"]["hits_at_10"]

            print(f"epoch: {epoch}  loss:{epoch_loss}")
            print(f"mmr:{mmr}, mr:{mr}, hits1:{hits1}, hits5:{hits5}, hits10:{hits10}")
            print("")
            self.result_tracker.log_metrics(metrics=result.to_flat_dict(), step=epoch, prefix=self.prefix)
        if epoch >= 1 and epoch % self.full_test_frequency == 0:
            print("")
            print("full_test ........")
            result = self.evaluator.evaluate(
                use_tqdm = False,
                additional_filter_triples=self.additional_filter_triples,
                model=self.model,
                mapped_triples=self.full_test_evaluation_triples,
                device=self.training_loop.device,
                batch_size=self.evaluator.batch_size or self.batch_size,
                **self.kwargs,
            )
            mmr = result.to_dict()["both"]["optimistic"]["inverse_harmonic_mean_rank"]
            mr = result.to_dict()["both"]["optimistic"]["arithmetic_mean_rank"]
            hits1 = result.to_dict()["both"]["optimistic"]["hits_at_1"]
            hits5 = result.to_dict()["both"]["optimistic"]["hits_at_5"]
            hits10 = result.to_dict()["both"]["optimistic"]["hits_at_10"]
            print(f"epoch: {epoch}  loss:{epoch_loss}")
            print(f"mmr:{mmr}, mr:{mr}, hits1:{hits1}, hits5:{hits5}, hits10:{hits10}")
            print("")
            self.result_tracker.log_metrics(metrics=result.to_flat_dict(), step=epoch, prefix=self.prefix)
