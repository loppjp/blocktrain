from dataclasses import dataclass

import torch
from tqdm import tqdm

from blocktrain.component_api import IComponentProvider
from blocktrain.trainer_api import ITrainer
from blocktrain.callbacks import (
    ICallback,
    EpochTrainingOutputData,
    TrainingStepInputData,
    TrainingStepOutputData,
    EvalStepInputData,
    EvalStepOutputData,
)
from blocktrain.experiment_api import (
    TrainingInputParameters,
    EpochInputData,
)


class BaseTrainer(ITrainer):

    def __init__(
        self,
        *args,
        training_input_parameters: dict = None,
        component_provider: IComponentProvider = None,
        **kwargs,
    ):
        self.training_input_parameters: TrainingInputParameters = (
            training_input_parameters
        )

        self.callbacks: list[ICallback]

        self.component_provider: IComponentProvider = component_provider

    def train(self):

        for callback in self.callbacks:
            callback.on_training_start()

        for epoch_idx in tqdm(range(self.training_input_parameters.num_epochs)):

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

        self.eval_epoch()

        for callback in self.callbacks:
            callback.on_eval_end()

    def train_epoch(self):
        pass

    def eval_epoch(self):
        pass

    def set_component_provider(self, component_provider):
        self.component_provider = component_provider


class Trainer(BaseTrainer):

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

    def train_epoch(self, epoch_data: EpochInputData):

        model: torch.nn = self.component_provider.get_model()
        model.train(True)
        optimizer: torch.optim.Optimizer = self.component_provider.get_optimizer()
        loss_function: torch.Tensor = self.component_provider.get_loss_function()

        for idx, data in enumerate(epoch_data.dataloader):

            for callback in self.callbacks:
                callback.on_train_step_start(TrainingStepInputData(step_number=idx))

            X, y = data
            optimizer.zero_grad()
            losses = model(X, y)
            losses_reduced = sum(loss for loss in losses.values())
            losses = losses_reduced.item()
            losses.backward()
            optimizer.step()

            for callback in self.callbacks:
                callback.on_train_step_end(TrainingStepOutputData(loss=losses.item()))

    def eval_epoch(self, epoch_data: EpochInputData):

        model: torch.nn = self.component_provider.get_model()
        model.eval()
        loss_function: torch.Tensor = self.component_provider.get_loss_function()

        with torch.no_grad():

            for idx, data in enumerate(epoch_data.dataloader):

                for callback in self.callbacks:
                    callback.on_eval_step_start(
                        EvalStepInputData(
                            step_number=idx,
                        )
                    )

                X, y = data
                y_hat = model(X)
                loss = loss_function(y_hat, y)
                loss.backward()

                for callback in self.callbacks:
                    callback.on_eval_step_end(EvalStepOutputData(loss=loss.item()))
