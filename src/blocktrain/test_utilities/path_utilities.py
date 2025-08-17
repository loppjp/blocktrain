from pathlib import Path

import blocktrain

def get_config_path():
    """
    This function makes a relative path assumption about the location
    of the config directory for experiments
    """
    return Path(blocktrain.__file__).parent.parent.parent/"config"

def get_test_experiment():
    return get_config_path()/"test_experiment.yaml"