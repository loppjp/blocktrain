from pathlib import Path

from blocktrain.factories.experiment_factory import from_file
from blocktrain.experiment_api import BaseExperiment
from blocktrain.utilities import get_config_path

DEFAULT_CONFIG_PATH: Path = get_config_path()/"experiment.yaml"

def main(
    experiment_path: Path = None,
    **kwargs
):
    if experiment_path is None:
        experiment_path = DEFAULT_CONFIG_PATH
    experiment: BaseExperiment = from_file(experiment_path)

    experiment.get_trainer().train()

if __name__ == "__main__":
    main()