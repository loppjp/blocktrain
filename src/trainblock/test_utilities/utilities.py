from pathlib import Path

import trainblock
from trainblock.utilities import get_config_path

def get_test_experiment():
    return get_config_path()/"test_experiment.yaml"