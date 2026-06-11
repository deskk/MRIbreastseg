import os
import json
import logging
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_crop_metadata(json_path: str) -> dict:
    """Loads the serialized affine physics metadata for a given cropped region."""
    logging.info(f"Loading metadata coordinates from {json_path}")
    with open(json_path, 'r') as jf:
        metadata = json.load(jf)
    return metadata

def reconstitute_to_full_torso(full_torso_reference: sitk.Image, 
                               mask_1x1x1_path: str, 
                               metadata: dict) -> sitk.Image:
    """
    Reverse registration pipeline. Assigns the previously extracted Origin, Direction, 
    and 1.0mm assumed spacing to the MAMA-MIA prediction to reorient it absolute physically,
    and then applies sitk.ResampleImageFilter against the original baseline reference to natively 
    snap the crop back into the holistic geometry space natively.
    
    Rule 4: Reconstitution via Physical Space using Nearest Neighbor.
    """
    logging.info(f"Loading prediction mask: {mask_1x1x1_path}")
    pred_mask = sitk.ReadImage(mask_1x1x1_path)
    
    # 1. Reverse Affine Declaration: Force the floating prediction back into physical lockstep.
    # We assign the original crop's origin and direction to mathematically anchor it.
    pred_mask.SetOrigin(metadata["Origin"])
    pred_mask.SetDirection(metadata["Direction"])
    # By strictly forcing 1x1x1 spacing, alongside the old Origin, we define exactly where 
    # the 1.0mm inference geometry resides relative to the scanner's (0,0,0) world coordinates.
    pred_mask.SetSpacing((1.0, 1.0, 1.0))
    
    # Ensure standard mask datatype
    pred_mask = sitk.Cast(pred_mask, sitk.sitkUInt8)
    
    # 2. Resample mapping onto the absolute full-torso coordinate graph
    logging.info("Resampling prediction mask back to original full-torso geometric space...")
    resampler = sitk.ResampleImageFilter()
    # The reference image natively dictates output Size, OutputSpacing, OutputOrigin, OutputDirection
    resampler.SetReferenceImage(full_torso_reference)
    
    # Crucial: Nearest Neighbor to preserve categorical labels organically without interpolation ghosts
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0) # Blank canvas default
    
    reconstituted_mask = resampler.Execute(pred_mask)
    return reconstituted_mask

def process_recombination(original_full_torso_path: str, 
                          left_mask_path: str, 
                          left_metadata_path: str,
                          right_mask_path: str, 
                          right_metadata_path: str,
                          output_combined_mask_path: str):
    """
    Execution bridge. Orchestrates left and right segmentations sequentially and coalesces them.
    """
    logging.info(f"--- Starting Recombination Pipeline ---")
    logging.info(f"Reference Scan: {original_full_torso_path}")
    
    reference_img = sitk.ReadImage(original_full_torso_path)
    
    # Reconstitute Left Hemisphere
    left_meta = load_crop_metadata(left_metadata_path)
    left_reconstituted = reconstitute_to_full_torso(reference_img, left_mask_path, left_meta)
    
    # Reconstitute Right Hemisphere
    right_meta = load_crop_metadata(right_metadata_path)
    right_reconstituted = reconstitute_to_full_torso(reference_img, right_mask_path, right_meta)
    
    # Coalesce via Maximum logic: 
    # Left and Right segmentations ideally are non-overlapping in world geometry. 
    # sitk.Maximum cleanly combines two blank arrays populated with isolated label structures.
    logging.info("Coalescing Left and Right masks spatially...")
    combined_mask = sitk.Maximum(left_reconstituted, right_reconstituted)
    
    sitk.WriteImage(combined_mask, output_combined_mask_path)
    logging.info(f"Fully assembled reconstituted full-torso mask saved to: {output_combined_mask_path}")

if __name__ == "__main__":
    pass
