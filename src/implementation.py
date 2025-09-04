import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent

sys.path.append(str(parent_dir))

from shared.shared_functions import *

def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.split(' ', 1)
                config[key] = value.strip()
    return config


