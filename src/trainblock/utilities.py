from pathlib import Path

import trainblock

def get_config_path():
    """
    This function makes a relative path assumption about the location
    of the config directory for experiments
    """
    return Path(trainblock.__file__).parent.parent.parent/"config"