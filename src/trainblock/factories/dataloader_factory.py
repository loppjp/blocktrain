from torch.utils.data import Dataset, DataLoader 

from trainblock.factories.loader import load

def dataloader_factory(dataset: Dataset, **kwargs) -> DataLoader:
    """
    The dataloader factory requires a torch Dataset to provide to the instantiated
    DataLoader. Leverage the feature of the load function to provide the dataset
    in __args__

    Args:
        dataset: the dataset for this Dataloader to use at runtime
        kwargs: Any keyword arguments to be forwarded to the Dataloader upon
                instantiation
    """

    return load({
        "__args__":[dataset],
        **kwargs,
    })