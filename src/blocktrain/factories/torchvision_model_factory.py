from blocktrain.factories.loader import load


def load_torchvision_model(
    model_args=[],
    model_kwargs={},
    torchvision_class=None,
    torchvision_module="torchvision.models",
):
    return load(
        {
            "__args__": model_args,
            "__module__": torchvision_module,
            "__class__": torchvision_class,
            **model_kwargs,
        }
    )
