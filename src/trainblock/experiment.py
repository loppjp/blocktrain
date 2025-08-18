from trainblock.factories.loader import load
from trainblock.factories.dataloader_factory import dataloader_factory
from trainblock.experiment_api import BaseExperiment


class Experiment(BaseExperiment):
    """
    A class that governs an instance of an experiment
    """
    def __init__(
        self,
        train_dataset=None,
        eval_dataset=None,
        callbacks=None,
        model=None,
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.callbacks = callbacks
        self.model = model

        # dataloaders
        self.train_dataloader = None
        self.eval_dataloader = None
        
        # optimizer (requires model weights)
        self.optimizer = None

        self.collator = None

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_train_dataset(self):
        return self.train_dataset

    def get_eval_dataset(self):
        return self.eval_dataset

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_eval_dataloader(self):
        return self.eval_dataloader

    def get_collator(self):
        return self.collator


def experiment_factory(*_, **kwargs) -> Experiment:
    """
    Construct an experiment from 
    """
    e = Experiment(
        train_dataset=load(kwargs["train_dataset"]),
        eval_dataset=load(kwargs["eval_dataset"]),
        callbacks=[load(cb) for cb in kwargs["callbacks"]],
        model=load(kwargs["model"]),
    )

    # pass self as a IComponentProvider
    e.trainer = load(kwargs["trainer"], component_provider=e)

    e.optimizer = load(kwargs["optimizer"], component_provider=e)

    e.collator = load(kwargs["collator"])

    e.train_dataloader = dataloader_factory(
        e.get_train_dataset(),
        collate_fn=e.collator,
        **kwargs["train_dataloader"]
    )

    e.eval_dataloader = dataloader_factory(
        e.get_eval_dataset(),
        collate_fn=e.collator,
        **kwargs["eval_dataloader"]
    )

    return e