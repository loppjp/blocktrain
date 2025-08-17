import torch

from trainblock.factories.loader import load
from trainblock.component_api import IComponentProvider

def load_optimizer(
    *args,
    optimizer_module:str = "torch.optim",
    optimizer_class:str = None,
    component_provider: IComponentProvider = None,
    **kwargs,
):
    """
    The torch optimizers require parametres. Therefore,
    this factory will support access to the model by leveraging
    the component_provider interface
    """
    model: torch.nn = component_provider.get_model()
    return load(
        {
            "__module__":optimizer_module,
            "__class__":optimizer_class,
        },
        model.parameters(), 
        **kwargs
    )