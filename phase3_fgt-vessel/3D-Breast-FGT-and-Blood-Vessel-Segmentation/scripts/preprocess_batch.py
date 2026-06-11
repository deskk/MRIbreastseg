import os
import glob
import sys
import numpy as np
import SimpleITK as sitk
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import normalize_image, zscore_image


import json
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)
config = load_config()


def main():
    phase1_masks_dir = config["PHASE1"]["OUTPUT_MASK_DIR"]:
    main()
