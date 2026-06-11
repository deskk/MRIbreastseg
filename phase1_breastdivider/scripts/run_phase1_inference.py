import os
import glob
import random
import shutil
import tempfile
import subprocess


import json
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)
config = load_config()


INPUT_DIR = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]:
    main()
