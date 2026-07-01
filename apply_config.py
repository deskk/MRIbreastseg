import os
import re

CONFIG_IMPORT = """
import json
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '{rel_path}config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)
config = load_config()
"""

FILES = [
    {
        "path": "pipeline/phase1_breastdivider/scripts/run_phase1_inference.py",
        "rel": "../../../",
        "replacements": [
            (r'INPUT_DIR = ".*"', 'INPUT_DIR = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]'),
            (r'OUTPUT_DIR = ".*"', 'OUTPUT_DIR = config["PHASE1"]["OUTPUT_MASK_DIR"]'),
            (r'MODEL_DIR = ".*"', 'MODEL_DIR = config["PHASE1"]["MODEL_DIR"]')
        ]
    },
    {
        "path": "pipeline/phase1_breastdivider/scripts/split_breasts_midline.py",
        "rel": "../../../",
        "replacements": [
            (r'MASK_DIR = ".*"', 'MASK_DIR = config["PHASE1"]["OUTPUT_MASK_DIR"]'),
            (r'REGISTERED_DIR = ".*"', 'REGISTERED_DIR = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]'),
            (r'LEFT_OUT_DIR = ".*"', 'LEFT_OUT_DIR = config["PHASE1"]["OUTPUT_LEFT_DIR"]'),
            (r'RIGHT_OUT_DIR = ".*"', 'RIGHT_OUT_DIR = config["PHASE1"]["OUTPUT_RIGHT_DIR"]')
        ]
    },
    {
        "path": "pipeline/phase2_mama-mia/scripts/run_phase2_mamamia.py",
        "rel": "../../../",
        "replacements": [
            (r'INPUT_DIRS = \[.*?\]', 'INPUT_DIRS = [config["PHASE1"]["OUTPUT_LEFT_DIR"], config["PHASE1"]["OUTPUT_RIGHT_DIR"]]'),
            (r'OUTPUT_DIRS = \[.*?\]', 'OUTPUT_DIRS = [config["PHASE2"]["OUTPUT_LEFT_DIR"], config["PHASE2"]["OUTPUT_RIGHT_DIR"]]'),
            (r'env\["nnUNet_raw"\] = ".*"', 'env["nnUNet_raw"] = config["PHASE2"]["NNUNET_RAW"]'),
            (r'env\["nnUNet_preprocessed"\] = ".*"', 'env["nnUNet_preprocessed"] = config["PHASE2"]["NNUNET_PREPROCESSED"]'),
            (r'env\["nnUNet_results"\] = ".*"', 'env["nnUNet_results"] = config["PHASE2"]["NNUNET_RESULTS"]')
        ]
    },
    {
        "path": "pipeline/phase2_mama-mia/scripts/postprocess_filter_noise.py",
        "rel": "../../../",
        "replacements": [
            (r'OUTPUT_DIRS = \[.*?\]', 'OUTPUT_DIRS = [config["PHASE2"]["OUTPUT_LEFT_DIR"], config["PHASE2"]["OUTPUT_RIGHT_DIR"]]')
        ]
    },
    {
        "path": "pipeline/phase3_fgt-vessel/3D-Breast-FGT-and-Blood-Vessel-Segmentation/scripts/preprocess_batch.py",
        "rel": "../../../../",
        "replacements": [
            (r'phase1_masks_dir = ".*"', 'phase1_masks_dir = config["PHASE1"]["OUTPUT_MASK_DIR"]'),
            (r'registered_dir = ".*"', 'registered_dir = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]'),
            (r'out_dir = "inference_data/batch/images"', 'out_dir = config["PHASE3"]["INFERENCE_IMAGES_DIR"]'),
            (r'os\.path\.join\("/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/data/phase4_fusion/left"', 'os.path.join(config["PHASE5"]["OUTPUT_FUSION_LEFT_DIR"]')
        ]
    },
    {
        "path": "pipeline/phase3_fgt-vessel/3D-Breast-FGT-and-Blood-Vessel-Segmentation/scripts/export_nifti_batch.py",
        "rel": "../../../../",
        "replacements": [
            (r'registered_dir = ".*"', 'registered_dir = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]'),
            (r'preds_breast_dir = ".*"', 'preds_breast_dir = config["PHASE3"]["PREDS_BREAST_DIR"]'),
            (r'preds_dv_dir = ".*"', 'preds_dv_dir = config["PHASE3"]["PREDS_DV_DIR"]'),
            (r'out_dir = ".*"', 'out_dir = config["PHASE3"]["OUTPUT_FGT_VESSEL_DIR"]')
        ]
    },
    {
        "path": "pipeline/phase5_fusion/scripts/calculate_birads.py",
        "rel": "../../../",
        "replacements": [
            (r'base_data_dir = ".*"', 'base_data_dir = config["DATA_DIR"]'),
            (r'fgt_dir = os\.path\.join\(base_data_dir, "fgt-vessel_fulltorso"\)', 'fgt_dir = config["PHASE3"]["OUTPUT_FGT_VESSEL_DIR"]'),
            (r'tumor_csv_path = os\.path\.join\(base_data_dir, "tumor_presence.csv"\)', 'tumor_csv_path = config["PHASE5"]["OUTPUT_TUMOR_PRESENCE_CSV"]'),
            (r'output_csv_path = os\.path\.join\(base_data_dir, "tumor_and_birads.csv"\)', 'output_csv_path = config["PHASE5"]["OUTPUT_TUMOR_BIRADS_CSV"]')
        ]
    },
    {
        "path": "pipeline/phase5_fusion/scripts/generate_figures.py",
        "rel": "../../../",
        "replacements": [
            (r'base_data_dir = ".*"', 'base_data_dir = config["DATA_DIR"]'),
            (r'phase1_dir = os\.path\.join\(base_data_dir, "phase1_mask"\)', 'phase1_dir = config["PHASE1"]["OUTPUT_MASK_DIR"]'),
            (r'fusion_dir = os\.path\.join\(base_data_dir, "phase4_fusion"\)', 'fusion_dir = os.path.dirname(config["PHASE5"]["OUTPUT_FUSION_LEFT_DIR"])'),
            (r'figures_dir = os\.path\.join\(fusion_dir, "figures"\)', 'figures_dir = config["PHASE5"]["OUTPUT_FIGURES_DIR"]')
        ]
    }
]

for item in FILES:
    filepath = item["path"]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
        
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Insert config import after the last import statement
    import_match = list(re.finditer(r'^(?:import|from) .*\n', content, re.MULTILINE))
    if import_match:
        last_import = import_match[-1]
        insert_idx = last_import.end()
        content = content[:insert_idx] + "\n" + CONFIG_IMPORT.format(rel_path=item["rel"]) + "\n" + content[insert_idx:]
    else:
        content = CONFIG_IMPORT.format(rel_path=item["rel"]) + "\n" + content
        
    for pattern, replacement in item["replacements"]:
        content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.MULTILINE)
        
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Updated {filepath}")
