#!/bin/bash
cd phase3_fgt-vessel/3D-Breast-FGT-and-Blood-Vessel-Segmentation
echo "Running preprocess"
conda run -n fgt_env python scripts/preprocess_batch.py
echo "Running breast prediction"
conda run -n fgt_env python predict.py -c breast -i inference_data/batch/images -s inference_data/batch/preds_breast -p trained_models/breast_model.pth
echo "Running dv prediction"
conda run -n fgt_env python predict.py -c dv -i inference_data/batch/images -m inference_data/batch/preds_breast -s inference_data/batch/preds_dv -p trained_models/dv_model.pth
echo "Exporting NIfTI"
conda run -n fgt_env python scripts/export_nifti_batch.py
