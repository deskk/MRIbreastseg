import os
import glob
import numpy as np
import SimpleITK as sitk


import json
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)
config = load_config()


def inverse_transform(arr):
    # Reverse of np.rot90(arr, k=3, axes=(0, 1))
    arr = np.rot90(arr, k=1, axes=(0, 1))
    
    # Reverse of np.transpose(arr, (2, 1, 0)) -> This is its own inverse!
    arr = np.transpose(arr, (2, 1, 0))
    return arr

def main():
    phase1_masks_dir = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/data/phase1_mask"
    registered_dir = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]:
    main()
