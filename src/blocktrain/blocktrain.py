from pathlib import Path


class ITrainConfig:
    pass

class IEvalConfig:
    pass

class BlockTrainAPI:
    def train(self, config : ITrainConfig, *args, **kwargs):
        pass

    def eval(self, config : IEvalConfig, *args, **kwargs):
        pass



class BlockTrain(BlockTrainAPI):

    def __init__(self, *args, **kwargs):
        pass


class BlockTrainFactory:
    """
    Load blocktrain instances given the 
    """
    def from_experiment_file(experiment: Path) -> BlockTrain: