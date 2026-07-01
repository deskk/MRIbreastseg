import os
import glob
import subprocess
import shutil
import tempfile
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Constants
SUBJECTS = ['046699', '076900', '081567', '082396', '090124']
DATASET_PATH = "/local/scratch/scratch-hd/desmond/dataset/clean_data_registered"
BASE_DIR = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/phase2_mama-mia"
MARGIN_MM = 20.0

def get_pre_contrast_series(subject_dir):
    all_files = glob.glob(os.path.join(subject_dir, "*.nii.gz"))
    for f in all_files:
        filename = os.path.basename(f).lower()
        if "pre" in filename and "t1" in filename:
            return f
    return None

def run_breastdivider_inference(subject, img_path, mask_path):
    print(f"Running BreastDivider for {subject}...")
    with tempfile.TemporaryDirectory() as temp_in, tempfile.TemporaryDirectory() as temp_out:
        in_file = os.path.join(temp_in, f"{subject}_0000.nii.gz")
        shutil.copy(img_path, in_file)
        cmd = f"nnUNetv2_predict_from_modelfolder -i {temp_in} -o {temp_out} -m /local/scratch/scratch-hd/desmond/research/breastdivider/BreastDividerModel -f all"
        subprocess.run(cmd, shell=True, check=True)
        out_file = os.path.join(temp_out, f"{subject}.nii.gz")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        shutil.copy(out_file, mask_path)

def calculate_padded_bbox_voxels(mask_image, target_label, margin_mm):
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask_image)
    if not label_stats.HasLabel(target_label): return None
    bbox = label_stats.GetBoundingBox(target_label) # [x, y, z, dx, dy, dz]
    min_idx = list(bbox[:3])
    size = list(bbox[3:])
    max_idx = [min_idx[i] + size[i] - 1 for i in range(3)]
    
    img_size = mask_image.GetSize()
    spacing = mask_image.GetSpacing()
    
    padded_min = []
    padded_max = []
    
    for i in range(3):
        if i == 1:
            voxel_pad_max = int(np.ceil(margin_mm / spacing[i]))
            voxel_pad_min = int(np.ceil(10.0 / spacing[i]))
        else:
            voxel_pad_max = 0
            voxel_pad_min = 0
            
        padded_min.append(max(0, min_idx[i] - voxel_pad_min))
        padded_max.append(min(img_size[i] - 1, max_idx[i] + voxel_pad_max))
        
    return padded_min, padded_max

def draw_3d_bbox(ax, bbox, color):
    if not bbox: return
    min_v, max_v = bbox
    x0, y0, z0 = min_v
    x1, y1, z1 = max_v
    edges = [
        [(x0, y0, z0), (x1, y0, z0)],
        [(x1, y0, z0), (x1, y1, z0)],
        [(x1, y1, z0), (x0, y1, z0)],
        [(x0, y1, z0), (x0, y0, z0)],
        [(x0, y0, z1), (x1, y0, z1)],
        [(x1, y0, z1), (x1, y1, z1)],
        [(x1, y1, z1), (x0, y1, z1)],
        [(x0, y1, z1), (x0, y0, z1)],
        [(x0, y0, z0), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y0, z1)],
        [(x1, y1, z0), (x1, y1, z1)],
        [(x0, y1, z0), (x0, y1, z1)]
    ]
    for p1, p2 in edges:
        ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2)

def generate_viz():
    for subject in SUBJECTS:
        subj_dir = os.path.join(DATASET_PATH, subject)
        img_path = get_pre_contrast_series(subj_dir)
        if not img_path:
            print(f"Pre-contrast image not found for {subject}")
            continue

        mask_path = f"/local/scratch/scratch-hd/desmond/research/nnUNet_output/{subject}/{subject}_BreastDivider_Mask.nii.gz"
        
        try:
            run_breastdivider_inference(subject, img_path, mask_path)
        except Exception as e:
            print(f"Inference failed for {subject}: {e}")
            continue
            
        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)
        
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(mask)
        if not label_stats.HasLabel(1):
            print(f"Label 1 missing in {subject}")
            continue
            
        centroid_v = mask.TransformPhysicalPointToIndex(label_stats.GetCentroid(1)) 
        x_idx, y_idx, z_idx = centroid_v
        
        img_arr = sitk.GetArrayFromImage(img)
        mask_arr = sitk.GetArrayFromImage(mask)
        
        # Z is first dim in np array: (z, y, x)
        axial_img = img_arr[z_idx, :, :]
        axial_mask = mask_arr[z_idx, :, :]
        
        coronal_img = img_arr[:, y_idx, :]
        coronal_mask = mask_arr[:, y_idx, :]
        
        sagittal_img = img_arr[:, :, x_idx]
        sagittal_mask = mask_arr[:, :, x_idx]
        
        bbox_l = calculate_padded_bbox_voxels(mask, 1, MARGIN_MM)
        bbox_r = calculate_padded_bbox_voxels(mask, 2, MARGIN_MM)
        
        fig = plt.figure(figsize=(18, 12))
        
        # 3D View
        ax3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax3d.set_title("3D Bounding Boxes", fontsize=14)
        draw_3d_bbox(ax3d, bbox_l, 'cyan')
        draw_3d_bbox(ax3d, bbox_r, 'magenta')
        sz = img.GetSize()
        ax3d.set_xlim([0, sz[0]])
        ax3d.set_ylim([0, sz[1]])
        ax3d.set_zlim([0, sz[2]])
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        
        cmap_mask = ListedColormap(['none', 'cyan', 'magenta']) 
        
        # Helper to draw 2D bboxes
        def add_2d_rect(ax, bbox, plane, color, label):
            if not bbox: return
            min_v, max_v = bbox
            if plane == 'axial':
                # xy plane
                x, y = min_v[0], min_v[1]
                w, h = max_v[0] - x, max_v[1] - y
            elif plane == 'coronal':
                # xz plane -> array maps to (z, x)
                x, y = min_v[0], min_v[2]
                w, h = max_v[0] - x, max_v[2] - y
            elif plane == 'sagittal':
                # yz plane -> array maps to (z, y)
                x, y = min_v[1], min_v[2]
                w, h = max_v[1] - x, max_v[2] - y
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', label=label))

        # Axial
        ax_ax = fig.add_subplot(2, 2, 2)
        ax_ax.imshow(axial_img, cmap='gray')
        ax_ax.imshow(np.ma.masked_where(axial_mask == 0, axial_mask), cmap=cmap_mask, alpha=0.4)
        add_2d_rect(ax_ax, bbox_l, 'axial', 'cyan', 'Left Crop')
        add_2d_rect(ax_ax, bbox_r, 'axial', 'magenta', 'Right Crop')
        ax_ax.set_title(f"Axial Slice {z_idx}", fontsize=14)
        ax_ax.axis('off')
        
        # Coronal
        ax_cor = fig.add_subplot(2, 2, 3)
        ax_cor.imshow(coronal_img, cmap='gray')
        ax_cor.imshow(np.ma.masked_where(coronal_mask == 0, coronal_mask), cmap=cmap_mask, alpha=0.4)
        add_2d_rect(ax_cor, bbox_l, 'coronal', 'cyan', 'Left Crop')
        add_2d_rect(ax_cor, bbox_r, 'coronal', 'magenta', 'Right Crop')
        ax_cor.set_title(f"Coronal Slice {y_idx}", fontsize=14)
        ax_cor.axis('off')
        
        # Sagittal
        ax_sag = fig.add_subplot(2, 2, 4)
        ax_sag.imshow(sagittal_img, cmap='gray')
        ax_sag.imshow(np.ma.masked_where(sagittal_mask == 0, sagittal_mask), cmap=cmap_mask, alpha=0.4)
        add_2d_rect(ax_sag, bbox_l, 'sagittal', 'cyan', 'Left Crop')
        add_2d_rect(ax_sag, bbox_r, 'sagittal', 'magenta', 'Right Crop')
        ax_sag.set_title(f"Sagittal Slice {x_idx}", fontsize=14)
        ax_sag.axis('off')
        
        plt.suptitle(f"Methodology: BreastDivider Segmentation\nSubject {subject}", fontsize=16)
        
        # Legend
        handles, labels = ax_ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles[:2], labels[:2], loc='lower center', ncol=2, fontsize=12)

        out_path = os.path.join(BASE_DIR, f"presentation_figures/{subject}_bbox_comparison.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Methodology visualization saved to {out_path}")

if __name__ == "__main__":
    generate_viz()
