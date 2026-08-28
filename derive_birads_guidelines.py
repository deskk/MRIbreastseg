import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def get_majority(row):
    counts = Counter(row.values)
    return counts.most_common(1)[0][0]

def calculate_midpoint_thresholds(df, metric_col, classes=['a', 'b', 'c', 'd']):
    medians = df.groupby('Consensus_BIRADS')[metric_col].median()
    # Reindex to ensure order
    medians = medians.reindex(classes)
    
    thresholds = {}
    if pd.notna(medians['a']) and pd.notna(medians['b']):
        thresholds['a_b'] = (medians['a'] + medians['b']) / 2
    if pd.notna(medians['b']) and pd.notna(medians['c']):
        thresholds['b_c'] = (medians['b'] + medians['c']) / 2
    if pd.notna(medians['c']) and pd.notna(medians['d']):
        thresholds['c_d'] = (medians['c'] + medians['d']) / 2
        
    return thresholds, medians

def main():
    out_dir = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/paper_fig"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    excel_path = "/local/scratch/scratch-hd/desmond/datasets/DUKE-fgtvessels/Breast_Radiologist_Density_Assessments.xlsx"
    df_rads = pd.read_excel(excel_path)
    
    cols = ['Radiologist A', 'Radiologist B', 'Radiologist C']
    df_rads_clean = df_rads[['Subject_ID'] + cols].dropna().copy()
    df_rads_clean['Consensus_BIRADS'] = df_rads_clean[cols].apply(get_majority, axis=1)
    
    csv_path = "/sci-it/projects/sarang-lab/desmond/MRIbreastseg/birads/mip_tumor_and_birads.csv"
    df_fgt = pd.read_csv(csv_path)
    
    df_fgt['Left_Sagittal_Ratio'] = df_fgt['Left_Sagittal_Ratio'].fillna(0)
    df_fgt['Right_Sagittal_Ratio'] = df_fgt['Right_Sagittal_Ratio'].fillna(0)
    
    # Compute Both Metrics
    df_fgt['Max_Sagittal_Ratio'] = df_fgt[['Left_Sagittal_Ratio', 'Right_Sagittal_Ratio']].max(axis=1)
    df_fgt['Avg_Sagittal_Ratio'] = df_fgt[['Left_Sagittal_Ratio', 'Right_Sagittal_Ratio']].mean(axis=1)
    
    df_fgt = df_fgt.rename(columns={'Subject': 'Subject_ID'})
    df_merged = pd.merge(df_rads_clean, df_fgt[['Subject_ID', 'Max_Sagittal_Ratio', 'Avg_Sagittal_Ratio']], on='Subject_ID', how='inner')
    
    order = ['a', 'b', 'c', 'd']
    
    # Calculate thresholds for both
    max_thresh, max_medians = calculate_midpoint_thresholds(df_merged, 'Max_Sagittal_Ratio', order)
    avg_thresh, avg_medians = calculate_midpoint_thresholds(df_merged, 'Avg_Sagittal_Ratio', order)
    
    # 2. Side-by-Side Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics = [
        ('Max_Sagittal_Ratio', 'Maximum Sagittal MIP Ratio', max_thresh),
        ('Avg_Sagittal_Ratio', 'Average Sagittal MIP Ratio', avg_thresh)
    ]
    
    for ax, (col, title, thresholds) in zip(axes, metrics):
        sns.boxplot(x='Consensus_BIRADS', y=col, data=df_merged, order=order, ax=ax, color='lightgray', showfliers=False)
        sns.stripplot(x='Consensus_BIRADS', y=col, data=df_merged, order=order, ax=ax, alpha=0.7, jitter=True, size=7)
        
        # Plot Threshold Lines
        colors = ['red', 'orange', 'green']
        for i, (key, val) in enumerate(thresholds.items()):
            ax.axhline(y=val, color=colors[i], linestyle='--', label=f'Threshold {key.upper().replace("_", "-")}: {val:.3f}')
            
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Consensus BI-RADS Category", fontsize=12)
        ax.set_ylabel("Computed Sagittal Ratio", fontsize=12)
        ax.legend()
        
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "birads_max_vs_avg_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    
    # 3. Output threshold analysis
    thresholds_path = os.path.join(out_dir, "birads_guideline_thresholds.txt")
    with open(thresholds_path, "w") as f:
        f.write("--- Empirical BI-RADS Thresholds (Sagittal MIP Method) ---\n\n")
        f.write("MAXIMUM SAGITTAL RATIO:\n")
        for k, v in max_thresh.items():
            f.write(f"  {k.upper().replace('_', ' vs ')}: {v:.4f}\n")
            
        f.write("\nAVERAGE SAGITTAL RATIO:\n")
        for k, v in avg_thresh.items():
            f.write(f"  {k.upper().replace('_', ' vs ')}: {v:.4f}\n")
            
    print(f"Comparison plot saved to {plot_path}")
    print(f"Thresholds saved to {thresholds_path}")

if __name__ == "__main__":
    main()
