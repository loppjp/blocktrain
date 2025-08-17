from torch.utils.data import Dataset, DataLoader 

from blocktrain.factories.loader import load

def dataloader_factory(dataset: Dataset, **kwargs) -> DataLoader:
    return load(
        dataset,
        **kwargs,
    )