import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_run_metrics(folder_path):
    """This function reads the 'rounds.csv' file from the specified folder path, extracts the relevant metrics
    and creates a two-panel plot showing performance and fairness metrics over the rounds. The plot is saved into the same folder with a unique filename based on the folder structure.
    Input:
        folder_path (str): The path to the folder containing the 'rounds.csv' file."""
    # Standard path for the data file
    csv_path = os.path.join(folder_path, "rounds.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: rounds.csv not found in {folder_path}")
        return

    # 1. Generate the unique filename
    # Extracts 'FedMinMax1' and 'run_EOFedMinMax_acs_iid5_2026-02-21...'
    parent_dir = os.path.basename(os.path.dirname(folder_path))
    current_dir = os.path.basename(folder_path)
    
    # Clean the filename (remove or truncate the specific seconds/timestamp if desired)
    clean_name = f"{parent_dir}_{current_dir}"
    output_filename = f"{clean_name}.png"

    # 2. Load the data
    df = pd.read_csv(csv_path)

    # 3. Create the Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top Panel: Performance
    ax1.plot(df['round'], df['accuracy'], label='Accuracy', color='#1f77b4', marker='o', markersize=3)
    ax1.plot(df['round'], df['balanced_accuracy'], label='Balanced Accuracy', color='#ff7f0e', linestyle='--')
    ax1.set_ylabel('Performance Score')
    ax1.set_title(f"Metrics for: {clean_name}")
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Bottom Panel: Fairness
    ax2.plot(df['round'], df['statistical_parity'], label='Statistical Parity', color='#2ca02c')
    ax2.plot(df['round'], df['equalized_odds'], label='Equalized Odds', color='#d62728')
    ax2.set_ylabel('Fairness Gap (Lower is Fairer)')
    ax2.set_xlabel('Round')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    # 4. Save using the specific path name
    plt.savefig(f"logs/FedMinMax/FedMinMax1/run_FedMinMax_acs_iid10_2026-02-22_12-27-09/{output_filename}", dpi=300)
    plt.close() # Closes plot to free up memory if running in a loop
    print(f"Plot saved as: {output_filename}")

if __name__ == "__main__":
    # Example usage: plot the metrics for a specific run and dataset
    example_folder = "logs/FedMinMax/FedMinMax1/run_FedMinMax_acs_iid10_2026-02-22_12-27-09/"
    plot_run_metrics(example_folder)