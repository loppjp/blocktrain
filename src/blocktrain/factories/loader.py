import importlib
from typing import Any
from copy import deepcopy
from copy import copy

"""
Factory keywords determine which strings are given special treatment
"""
KEYWORDS = ["__module__", "__class__", "__args__"]


def load(spec: dict, *args, **kwargs) -> Any:
    """
    Given a specification dictionary (spec), load a class from a module and
    pass the desired arguments and keyword arguments as specified.

    For example:

    __module__: "blocktrain.datasets",
    __class__: "S3Datset",
    __args__:
        - "s3://mydataset/path"
        - "cloud dataset"
    __kwargs__:
        batch_size: 16
        shuffle: False

    will instantiate the fully do something like:

        from blocktrain.datasets import S3Dataset

        return S3Dataset(
            "s3://mydataset/path",
            "cloud dataset",
            batch_size=16,
            shuffle=False
        )
    """

    # get the positiona arguments from the spec
    if "__args__" in spec:
        args = list(args)
        args.extend(spec["__args__"])

    # the kwargs dict is just the spec but without the keywords and their values
    kwargs.update(copy(spec))

    # get rid of keywords
    for keyword in KEYWORDS:
        if keyword in kwargs:
            kwargs.pop(keyword)

    # module is required, check and raise if its not there
    if "__module__" not in spec:
        raise RuntimeError(f"spec is missing __module__ keyword, found: {spec.keys()}")

    # class is required, check and raise if its not there
    if "__class__" not in spec:
        raise RuntimeError(f"spec is missing __class__ keyword, found: {spec.keys()}")

    # dynamically load the class from module and pass the args and kwargs
    # from the spec
    return getattr(importlib.import_module(spec["__module__"]), spec["__class__"])(
        *args, **kwargs
    )
