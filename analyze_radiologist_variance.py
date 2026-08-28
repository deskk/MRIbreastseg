import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_variance(excel_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_excel(excel_path)
    
    # Extract only the relevant columns
    cols = ['Radiologist A', 'Radiologist B', 'Radiologist C']
    df_rads = df[cols].dropna()
    
    # Calculate Agreement
    agreements = []
    for _, row in df_rads.iterrows():
        counts = Counter(row.values)
        max_agree = max(counts.values())
        if max_agree == 3:
            agreements.append("All 3 Agree")
        elif max_agree == 2:
            agreements.append("2 Agree")
        else:
            agreements.append("None Agree")
            
    df_rads['Agreement'] = agreements
    
    # Plot 1: Agreement Frequency
    plt.figure(figsize=(8, 6))
    agreement_counts = df_rads['Agreement'].value_counts()
    sns.barplot(x=agreement_counts.index, y=agreement_counts.values, palette="Blues_d")
    plt.title("Radiologist Agreement Frequency (n=50)", fontsize=14)
    plt.ylabel("Number of Subjects", fontsize=12)
    plt.savefig(os.path.join(out_dir, "radiologist_agreement_freq.png"))
    plt.close()
    
    # Plot 2: Pairwise Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [('Radiologist A', 'Radiologist B'), 
             ('Radiologist A', 'Radiologist C'), 
             ('Radiologist B', 'Radiologist C')]
    
    categories = ['a', 'b', 'c', 'd']
    
    for ax, (r1, r2) in zip(axes, pairs):
        confusion = pd.crosstab(df_rads[r1], df_rads[r2])
        # Reindex to ensure all categories a,b,c,d are shown even if 0
        confusion = confusion.reindex(index=categories, columns=categories, fill_value=0)
        
        sns.heatmap(confusion, annot=True, cmap="YlGnBu", cbar=False, ax=ax, fmt='d')
        ax.set_title(f"{r1} vs {r2}")
        ax.set_xlabel(r2)
        ax.set_ylabel(r1)
        
    plt.suptitle("Pairwise Radiologist BI-RADS Density (a, b, c, d) Classifications", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "radiologist_confusion.png"))
    plt.close()
    
    # Print a summary to console
    print("\n--- Phase 1: Radiologist Agreement Summary ---")
    print(agreement_counts.to_string())
    print(f"\nTotal subjects evaluated: {len(df_rads)}")
    
if __name__ == "__main__":
    excel_path = "/local/scratch/scratch-hd/desmond/datasets/DUKE-fgtvessels/Breast_Radiologist_Density_Assessments.xlsx"
    out_dir = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/paper_fig/eda"
    analyze_variance(excel_path, out_dir)
