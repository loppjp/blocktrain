from pathlib import Path

COMPONENT_REGISTRY = {}


class ITrainConfig:
    pass

class IEvalConfig:
    pass

class BlockTrainAPI:
    def train(self, config : ITrainConfig, *args, **kwargs):
        pass

    def eval(self, config : IEvalConfig, *args, **kwargs):
        pass

    def save_weights(self, save_path: Path):
        pass

    def load_weights(self, load_path: Path):
        pass

    def save(self, save_path: Path):
        """
        Save the state of training
        """
        pass

    def load(self, load_path: Path):
        """
        Load the state of training
        """
        pass


class BlockTrain(BlockTrainAPI):

    def __init__(self, *args, **kwargs):
        pass

class BlockTrainFactory:
    """
    Load blocktrain instances given the 
    """
    def from_experiment_file(experiment: Path) -> BlockTrain: