from blocktrain.factories.loader import load
from blocktrain.factories.dataloader_factory import dataloader_factory
from blocktrain.experiment_api import BaseExperiment


class Experiment(BaseExperiment):
    """
    A class that governs an instance of an experiment
    """
    def __init__(
        self,
        train_dataset=None,
        eval_dataset=None,
        trainer=None,
        callbacks=None,
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.trainer = trainer
        self.callbacks = callbacks

        # dataloaders
        self.train_dataloader = None
        self.eval_dataloader = None


def experiment_factory(*_, **kwargs) -> Experiment:
    """
    Construct an experiment from 
    """
    e = Experiment(
        train_dataset=load(kwargs["train_dataset"]),
        eval_dataset=load(kwargs["eval_dataset"]),
        trainer=load(kwargs["trainer"]),
        callbacks=[load(cb) for cb in kwargs["callbacks"]],
    )

    e.train_dataloader = dataloader_factory(
        e.get_train_dataset(),
        **kwargs["train_dataloader"]
    )

    e.eval_dataloader = dataloader_factory(
        e.get_eval_dataset(),
        **kwargs["eval_dataloader"]
    )



    return e