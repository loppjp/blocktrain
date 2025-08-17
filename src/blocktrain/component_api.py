from typing import Callable

from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from blocktrain.trainer_api import ITrainer
from blocktrain.callbacks import ICallback


class IEvalComponentProvider:
    def get_eval_dataset(self) -> Dataset: pass

    def get_eval_dataloader(self) -> DataLoader: pass


class ITrainingComponentProvider:
    def get_optimizer(self) -> Optimizer: pass

    def get_train_dataset(self) -> Dataset: pass

    def get_train_dataloader(self) -> DataLoader: pass

    def get_trainer(self) -> ITrainer: pass

    def get_callbacks(self) -> list[ICallback]: pass

    def get_loss_function(self) -> Callable: pass

class IComponentProvider(
    ITrainingComponentProvider,
    IEvalComponentProvider
):
    def get_component(self, component_name): pass

    def set_data(self, key, value): pass

    def get_model(self): pass
