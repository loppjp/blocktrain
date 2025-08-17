import yaml 
from pathlib import Path

from trainblock.experiment import Experiment
from trainblock.factories.loader import load

def unpack_experiment(spec: dict):
    """
    The experiment keyword is special, unpack it
    """
    if "experiment" not in spec:
        raise RuntimeError(f"spec is missing experiment keyword, found: {spec.keys()}")

    return spec["experiment"]

def from_file(path_to_experiment : Path) -> Experiment:
    with path_to_experiment.open() as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return from_experiment_spec(data)

def from_experiment_spec(spec: dict) -> Experiment:
    """
    Load an instance of an experiment from a specification
    dictionary
    """
    return load(unpack_experiment(spec))