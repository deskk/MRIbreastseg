import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

def main():
    csv_path = '/sci-it/projects/sarang-lab/desmond/MRIbreastseg/birads/sagittal_utah/uu_mip_tumor_and_birads.csv'
    out_dir = '/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/paper_fig'
    artifact_dir = '/home/desmond/.gemini/antigravity-ide/brain/4483c341-c4f8-40dd-849a-99984da733db/images'
    
    df = pd.read_csv(csv_path)
    
    def categorize_ratio(val):
        if pd.isna(val): return np.nan
        if val < 0.2410: return 'A'
        elif val < 0.3160: return 'B'
        elif val < 0.4133: return 'C'
        else: return 'D'

    left_classes = df['Left_Sagittal_Ratio'].apply(categorize_ratio)
    right_classes = df['Right_Sagittal_Ratio'].apply(categorize_ratio)
    
    sag_all = pd.concat([left_classes, right_classes]).dropna()
    sag_counts = sag_all.value_counts().reindex(['A', 'B', 'C', 'D']).fillna(0)
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['font.size'] = 10
    
    labels = ['A', 'B', 'C', 'D']
    x = np.arange(len(labels))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    
    rects = ax.bar(x, sag_counts.values, width, label='2D Sagittal MIP', color='darkgray', edgecolor='black', hatch='\\\\')
    
    ax.set_xlabel('BI-RADS Category')
    ax.set_ylabel('Number of Breasts')
    ax.set_title('Utah Dataset Distribution (Sagittal MIP)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    max_y = sag_counts.max()
    ax.set_ylim(0, max_y * 1.2)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    plot_path = os.path.join(out_dir, 'utah_sagittal_distribution.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    shutil.copy2(plot_path, os.path.join(artifact_dir, 'utah_sagittal_distribution.png'))
    
    # Markdown update removed as it is handled by Walkthrough artifact
        
    print(f"Distribution plot saved to {plot_path}")

if __name__ == '__main__':
    main()
