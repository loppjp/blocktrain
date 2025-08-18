from dataclasses import dataclass

import torch
from tqdm import tqdm

from trainblock.component_api import IComponentProvider
from trainblock.trainer_api import ITrainer
from trainblock.callbacks import (
    ICallback,
    EpochTrainingOutputData,
    TrainingStepInputData,
    TrainingStepOutputData,
    EvalStepInputData,
    EvalStepOutputData,
)
from trainblock.experiment_api import (
    TrainingInputParameters,
    TrainingEpochInputData,
    EvalEpochInputData,
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
            TrainingInputParameters(**training_input_parameters)
        )

        self.callbacks: list[ICallback] = []

        self.component_provider: IComponentProvider = component_provider

    def train(self):

        for callback in self.callbacks:
            callback.on_training_start()

        for epoch_idx in tqdm(range(self.training_input_parameters.num_epochs)):

            for callback in self.callbacks:
                callback.on_train_epoch_start()

            self.train_epoch(
                TrainingEpochInputData(
                    epoch_number=epoch_idx + 1,
                    dataloader=self.component_provider.get_train_dataloader(),
                )
            )

            for callback in self.callbacks:
                callback.on_train_epoch_end()

        for callback in self.callbacks:
            callback.on_training_end()

    def eval(self):

        for callback in self.callbacks:
            callback.on_eval_start()

        self.eval_epoch(EvalEpochInputData(
            self.component_provider.get_eval_dataloader()
        ))

        for callback in self.callbacks:
            callback.on_eval_end()

    def train_epoch(self, epoch_data: TrainingEpochInputData):
        pass

    def eval_epoch(self, epoch_data: EvalEpochInputData):
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

    def train_epoch(self, epoch_data: TrainingEpochInputData):

        model: torch.nn = self.component_provider.get_model()
        model.train(True)
        optimizer: torch.optim.Optimizer = self.component_provider.get_optimizer()

        for idx, data in enumerate(epoch_data.dataloader):

            for callback in self.callbacks:
                callback.on_train_step_start(TrainingStepInputData(step_number=idx))

            X, y = data
            optimizer.zero_grad()
            loss_dict = model(X, y)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            losses.backward()
            optimizer.step()

            for callback in self.callbacks:
                callback.on_train_step_end(TrainingStepOutputData(loss=loss_value))

    def eval_epoch(self, epoch_data: EvalEpochInputData):

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
