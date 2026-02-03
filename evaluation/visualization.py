import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import sys
import os

# 1. 设置样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# ==========================================
# Configuration: Aliases & Mappings
# ==========================================
# Define lists for batch processing
MODELS = ["Qwen3-0.6B-Final", "Qwen3-1.7B-Final", "Qwen3-4B-Final"]
DATASETS = ["aime25", "aime24", "hmmt25"]

def visualize_comparison_batch(baseline_strategy, target_strategy, full_df, primary_param, secondary_param=None):
    """
    使用 FacetGrid 在一张图上绘制所有模型和数据集的对比。
    """
    # Ensure numeric for plotting (Including Std_Acc)
    for col in ['Cost', 'Acc', 'Std_Acc']:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
            
    # Clean data based on essential columns
    full_df = full_df.dropna(subset=['Cost', 'Acc'])
    if "Solver" in target_strategy:
        target_df = full_df[full_df['Solver'].str.contains(target_strategy, regex=False)].copy()
    else:
        target_df = full_df[full_df['Strategy'].str.contains(target_strategy, regex=False)].copy()
    base_df = full_df.drop(target_df.index)


    if target_df.empty:
        print("No target strategy data found.")
        return

    # 3. 绘图 (Relplot)
    g = sns.relplot(
        data=target_df,
        x='Cost',
        y='Acc',
        hue=primary_param,
        style=secondary_param,
        col='Dataset',
        row='Model',
        palette='Blues',
        s=60, # Reduced size for cleaner look
        edgecolor='black',
        alpha=0.9,
        kind='scatter',
        height=5,
        aspect=1.2,
        facet_kws={'sharex': False, 'sharey': False}
    )

    # Add Error Bars for Target
    def plot_errorbars(data, **kws):
        if 'Std_Acc' not in data.columns:
             return
        # Filter for valid std deviations
        clean_data = data[data['Std_Acc'] > 0]
        if clean_data.empty:
            return

        plt.errorbar(
            x=clean_data['Cost'],
            y=clean_data['Acc'],
            yerr=clean_data['Std_Acc'],
            fmt='none',      # No markers, just lines
            ecolor='gray',   # Neutral color for error bars
            elinewidth=1.5,
            capsize=3,
            alpha=0.6,
            zorder=0         # Behind the main scatter points
        )
    
    g.map_dataframe(plot_errorbars)

    # 4. 在每个 Facet 上画 Baseline (使用 map_dataframe)
    def plot_baseline_layer(data, **kws):
        # data is the subset of target_df for the current facet
        # We need to find the corresponding baseline data for the same Model/Dataset
        if data.empty:
            return
            
        current_model = data['Model'].iloc[0]
        current_dataset = data['Dataset'].iloc[0]
        
        # Filter from GLOBAL full_df (captured from closure)
        base_subset = base_df[
            (base_df['Model'] == current_model) & 
            (base_df['Dataset'] == current_dataset)
        ].sort_values(by='Cost')
        
        if not base_subset.empty:
            ax = plt.gca()
            # Plot baseline line
            ax.plot(
                base_subset['Cost'], 
                base_subset['Acc'], 
                marker='o', markersize=4, linestyle='--', color='#555555', zorder=1, label='Baseline'
            )
            
            # Plot baseline error bars if available
            if 'Std_Acc' in base_subset.columns and base_subset['Std_Acc'].sum() > 0:
                ax.errorbar(
                    base_subset['Cost'],
                    base_subset['Acc'],
                    yerr=base_subset['Std_Acc'],
                    fmt='none',
                    ecolor='#555555',
                    elinewidth=1.5,
                    capsize=3,
                    alpha=0.6,
                    zorder=0
                )

            # Ensure log scale and adjust limits to show baseline
            ax.set_xscale('log')
            
            # Expand limits if baseline is outside current view
            x_min, x_max = ax.get_xlim()
            b_min, b_max = base_subset['Cost'].min(), base_subset['Cost'].max()
            ax.set_xlim(min(x_min, b_min * 0.9), max(x_max, b_max * 1.1))

    # Apply the baseline plotting function to each facet
    g.map_dataframe(plot_baseline_layer)

    # 5. 通用设置
    g.set_axis_labels("Token Cost (Log Scale)", "Accuracy (%)")
    
    # Enforce logarithmic scale with minor ticks/grid for all facets
    for ax in g.axes.flatten():
        ax.set_xscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.3)

    g.tight_layout()
    
    return g

def main():
    parser = argparse.ArgumentParser(description="Visualize Strategy Comparison")
    parser.add_argument("--baseline", default="FullReadStrategy", help="Baseline strategy substring")
    parser.add_argument("--target", default="ParallelDualWindow", help="Target strategy substring")
    
    # Updated Arguments
    parser.add_argument("--tune", default="T", help="Primary parameter to tune (Color)")
    parser.add_argument("--tune2", default=None, help="Secondary parameter to tune (Shape)")
    
    args = parser.parse_args()

    all_data_frames = []
    
    # Load all combinations
    for model in MODELS:
        for dataset in DATASETS:
            csv_path = f"matrix_results_{model}/{dataset}_raw.csv"
            if os.path.exists(csv_path):
                print(f"Loading {csv_path}...")
                df = pd.read_csv(csv_path)
                df['Model'] = model
                df['Dataset'] = dataset
                all_data_frames.append(df)
            else:
                print(f"Warning: {csv_path} not found.")

    full_df = pd.concat(all_data_frames, ignore_index=True)

    # Visualize
    g = visualize_comparison_batch(args.baseline, args.target, full_df, args.tune, args.tune2)

    # Save
    tune2_part = "" if args.tune2 is None else args.tune2
    out_name = f"img/{args.target}_by_{args.tune}_{tune2_part}.png"

    g.savefig(out_name, dpi=300)
    print(f"Visualization saved to {out_name}")

if __name__ == "__main__":
    main()