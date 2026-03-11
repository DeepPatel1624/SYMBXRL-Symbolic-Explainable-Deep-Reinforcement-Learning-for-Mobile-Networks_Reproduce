import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

# PROJ_ADDR = '/Users/deep/Desktop/MICxN/Projects/SYMBXRL-Symbolic-Explainable-Deep-Reinforcement-Learning-for-Mobile-Networks_Reproduce/MIMO_Scheduler'

KPI_LIST = ['MSEUr', 'DTUr']
QUARTILE_LIST = [f'Q{i}' for i in range(1, 5)] + ['MAX']

def create_effects_list_for_mean(kpis=KPI_LIST, changes=['dec', 'const', 'inc'], quartiles=QUARTILE_LIST):
    return {
        kpi: [f'{change}({kpi}, {quartile})' for quartile in quartiles for change in changes] for kpi in kpis
    }

def main():
    data_dir = os.path.join(script_dir, "Agents_Numeric_Symbolic_Raw_Data")
    csv_file = os.path.join(data_dir, "DS_LOS_HS2-Agent-SACG_HS2_1000-AS_No.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: Data file not found at {csv_file}")
        print("Please ensure generate_plot_data.py has been run successfully.")
        return
        
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    plt.rcParams.update({'font.size': 14})
    effects_list = create_effects_list_for_mean()
    
    # Plotting MSE Probability Distribution
    print("Generating MSE Probability Distribution Plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('MSE Probability Distribution (LOS High Speed)', fontsize=20)
    
    width = 0.5
    x = np.arange(len(effects_list['MSEUr']))
    effect_counts = df['MSEUr'].value_counts(normalize=True).reindex(effects_list['MSEUr'], fill_value=0)
    
    ax.bar(x, effect_counts.values, width, label='High Speed', color='#ff7f0e')
    
    ax.set_xlabel('Effect', fontsize=16)
    ax.set_ylabel('Probability', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(effects_list['MSEUr'], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    output_dir = os.path.join(script_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "MSE_Probability_Distribution_LOS_HS.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {output_path}")
    
    # Generate DTUr Graph as well if "DTUr" is in the columns
    if "DTUr" in df.columns:
        print("Generating DTUr Probability Distribution Plot...")
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        fig2.suptitle('DTUr Probability Distribution (LOS High Speed)', fontsize=20)
        
        x2 = np.arange(len(effects_list['DTUr']))
        effect_counts2 = df['DTUr'].value_counts(normalize=True).reindex(effects_list['DTUr'], fill_value=0)
        
        ax2.bar(x2, effect_counts2.values, width, label='High Speed', color='#1f77b4')
        
        ax2.set_xlabel('Effect', fontsize=16)
        ax2.set_ylabel('Probability', fontsize=16)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(effects_list['DTUr'], rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        fig2.subplots_adjust(top=0.9)
        
        output_path2 = os.path.join(output_dir, "DTUr_Probability_Distribution_LOS_HS.png")
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {output_path2}")

    if "decision" in df.columns:
        print("Generating Action Distribution Plot...")
        fig3, ax3 = plt.subplots(figsize=(14, 8))
        fig3.suptitle('Action (Decision) Distribution (LOS High Speed)', fontsize=20)
        
        # Calculate decision frequencies
        decision_counts = df['decision'].value_counts()
        x3 = np.arange(len(decision_counts))
        
        ax3.bar(x3, decision_counts.values, width, label='Frequency', color='#2ca02c')
        
        ax3.set_xlabel('Decision Categories', fontsize=16)
        ax3.set_ylabel('Count', fontsize=16)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(decision_counts.index, rotation=90, ha='center', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        fig3.subplots_adjust(top=0.9, bottom=0.25)
        
        output_path3 = os.path.join(output_dir, "Action_Distribution_LOS_HS.png")
        plt.savefig(output_path3, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {output_path3}")

    if "reward" in df.columns and "timestep" in df.columns:
        print("Generating Reward Over Time Plot...")
        fig4, ax4 = plt.subplots(figsize=(14, 8))
        fig4.suptitle('Reward Over Time (LOS High Speed)', fontsize=20)
        
        # Sort by timestep just in case
        df_sorted = df.sort_values(by='timestep')
        
        # Plot moving average to smooth out noise (window=10)
        window = min(10, len(df_sorted))
        moving_avg = df_sorted['reward'].rolling(window=window, min_periods=1).mean()
        
        ax4.plot(df_sorted['timestep'], df_sorted['reward'], alpha=0.3, color='#d62728', label='Instantaneous Reward')
        ax4.plot(df_sorted['timestep'], moving_avg, color='#d62728', linewidth=2, label=f'Moving Average (window={window})')
        
        ax4.set_xlabel('Timestep', fontsize=16)
        ax4.set_ylabel('Reward', fontsize=16)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        fig4.subplots_adjust(top=0.9)
        
        output_path4 = os.path.join(output_dir, "Reward_Over_Time_LOS_HS.png")
        plt.savefig(output_path4, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {output_path4}")

if __name__ == "__main__":
    main()
