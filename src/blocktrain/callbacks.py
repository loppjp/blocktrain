from dataclasses import dataclass

"""
This module includes the callback API (ICallback) and the data
sent to a registered callback function from the training
loop when the event occurs.
"""

@dataclass
class EpochTrainingOutputData:
    losses: float


@dataclass
class TrainingStepInputData:
    step_number: int


@dataclass
class TrainingStepOutputData:
    loss: float

@dataclass
class TrainingEpochInputData:
    epoch_number: int
    total_steps: int

@dataclass
class TrainingEpochOutputData:
    epoch_number: int
    total_steps: int

@dataclass
class EvalStepInputData:
    step_number: int


@dataclass
class EvalStepOutputData:
    loss: float


class ICallback:
    """
    Callbacks are to be executed upon occurance of the event
    that they describe
    """

    def on_train_step_start(self, data: TrainingStepInputData, *args, **kwargs): pass
    def on_train_step_end(self, data: TrainingStepOutputData, *args, **kwargs): pass

    def on_train_epoch_start(self, data:TrainingEpochInputData, *args, **kwargs): pass
    def on_train_epoch_end(self, data:TrainingEpochOutputData, *args, **kwargs): pass

    def on_eval_step_start(self, data:EvalStepInputData, *args, **kwargs): pass
    def on_eval_step_end(self, data:EvalStepOutputData, *args, **kwargs): pass

    def on_training_start(self): pass
    def on_training_end(self): pass

    def on_eval_start(self): pass
    def on_eval_end(self): pass