from blocktrain.factories.loader import load


class Experiment:
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


def experiment_factory(*_, **kwargs) -> Experiment:
    """
    Construct an experiment from 
    """
    return Experiment(
        train_dataset=load(kwargs["train_dataset"]),
        eval_dataset=load(kwargs["eval_dataset"]),
        trainer=load(kwargs["trainer"]),
        callbacks=[load(cb) for cb in kwargs["callbacks"]],
    )
