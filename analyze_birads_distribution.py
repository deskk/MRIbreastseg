import pandas as pd
import matplotlib.pyplot as plt

def analyze_birads(csv_path, dataset_name):
    print(f"\n--- Analysis for {dataset_name} ---")
    try:
        df = pd.read_csv(csv_path)
        
        if 'Left_BIRADS' in df.columns:
            left_col = 'Left_BIRADS'
            right_col = 'Right_BIRADS'
        elif 'Left_BiRADS' in df.columns:
            left_col = 'Left_BiRADS'
            right_col = 'Right_BiRADS'
        else:
            print("No BIRADS columns found.")
            return

        print("Left Breast Density Distribution:")
        print(df[left_col].value_counts(dropna=False).to_string())
        
        print("\nRight Breast Density Distribution:")
        print(df[right_col].value_counts(dropna=False).to_string())
        
        print("\nCombined (Both Breasts):")
        combined = pd.concat([df[left_col], df[right_col]])
        print(combined.value_counts(dropna=False).to_string())
        print(f"Total breasts analyzed: {len(combined)}")
        
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")

if __name__ == "__main__":
    analyze_birads('data-uu/tumor_and_birads.csv', 'data-uu Dataset')
    analyze_birads('external-duke-full/duke_outputs/tumor_and_birads.csv', 'external-duke-full Dataset')
    analyze_birads('external-duke-fgt/test_outputs/tumor_and_birads.csv', 'external-duke-fgt Dataset')
