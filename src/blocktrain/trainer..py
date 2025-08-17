from dataclasses import dataclass

import torch

from blocktrain.component_api import IComponentProvider
from blocktrain.trainer_api import ITrainer
from blocktrain.callbacks import ICallback
from blocktrain.experiment_api import (
    BaseExperiment, 
    TrainingInputParameters,
    EpochInputData,
    EpochTrainingOutputData,
    TrainingStepInputData,
    TrainingStepOutputData
)

class BaseTrainer(ITrainer):

    def __init__(
        self,
        *args,
        training_input_parameters: dict = None,
        component_provider: IComponentProvider = None,
        **kwargs,
    ):
        self.training_input_parameters: TrainingInputParameters = training_input_parameters
        
        self.callbacks: list[ICallback]

        self.component_provider: IComponentProvider = component_provider

    def train(self):

        for callback in self.callbacks:
            callback.on_training_start()

        for epoch_idx in range(self.training_input_parameters.num_epochs):

            for callback in self.callbacks:
                callback.on_train_epoch_start()

            self.train_epoch()

            for callback in self.callbacks:
                callback.on_train_epoch_end()

        for callback in self.callbacks:
            callback.on_training_end()


    def eval(self):

        for callback in self.callbacks:
            callback.on_eval_start()

        for callback in self.callbacks:
            callback.on_eval_end()

    def train_epoch(self): pass

    def set_component_provider(self, component_provider):
        self.component_provider = component_provider


class Trainer(BaseTrainer):

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super(*args, **kwargs)


    def train_epoch(self, epoch_data: EpochInputData):
        
        model: torch.nn = self.component_provider.get_model()
        model.train(True)

        for idx, data in enumerate(epoch_data.dataloader):
            
            for callback in self.callbacks:
                callback.on_train_step_start()

            X, y = data

            optimizer: torch.optim.Optimizer = self.component_provider.get_optimizer().zero_grad()

            y_hat = model(X)

            loss: torch.Tensor = self.component_provider.get_loss_function()(y_hat, y)
            loss.backward()

            optimizer.step()

            for callback in self.callbacks:
                callback.on_train_step_end()


def trainer_factory(
    *args,
    training_input_parameters: dict = None,
    **kwargs,
) -> ITrainer:
    pass