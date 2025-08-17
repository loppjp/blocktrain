from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset

from trainblock.callbacks import ICallback
from trainblock.trainer_api import ITrainer
from trainblock.component_api import IComponentProvider


@dataclass
class TrainingInputParameters:
    num_epochs: int


@dataclass
class TrainingEpochInputData:
    epoch_number: int
    dataloader: DataLoader

@dataclass
class EvalEpochInputData:
    dataloader: DataLoader


class BaseExperiment(IComponentProvider):
    def __init__(self, *args, **kwargs):
        self.train_dataset: Dataset = None
        self.eval_dataset: Dataset = None

        self.train_dataloader: DataLoader = None
        self.eval_dataloader: DataLoader = None

        self.callbacks: list[ICallback] = []

        self.trainer: ITrainer = None

        self.data: dict = {}
        self.components: dict = {}


    def get_train_dataset(self) -> Dataset:
        return self.train_dataset

    def get_eval_dataset(self) -> Dataset:
        return self.eval_dataset

    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def get_trainer(self) -> ITrainer:
        return self.trainer

    def get_callbacks(self) -> list[ICallback]:
        return self.callbacks

    def get_component(self, component_name):
        return self.components[component_name]

    def set_data(self, key, value):
        self.data[key] = value
