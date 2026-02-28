import os
import pandas as pd
import re

def aggregate_benchmarks(root_dir="logs"):
    """This function traverses the specified root directory to find all 'rounds.csv' files, extracts the final round metrics (round 51) 
    for each dataset and run, and aggregates these metrics by dataset to compute average performance and fairness scores. 
    The resulting DataFrame includes the average accuracy, balanced accuracy, statistical parity, equalized odds, and the sample size (number of runs) for each dataset.
    Input:
        root_dir (str): The root directory containing the benchmark run folders. Default is 'logs'.
    """
    all_observations = []

    for run_folder in os.listdir(root_dir):
        run_path = os.path.join(root_dir, run_folder)
        if not os.path.isdir(run_path):
            continue

        for ds_folder in os.listdir(run_path):

            match = re.search(r'run_Global_Group_Fairness_SP_(.*?)_\d{4}-\d{2}-\d{2}', ds_folder)
            
            if not match:
                match = re.search(r'run_Global_Group_Fairness_Eodd_(.*?)_\d{4}-\d{2}-\d{2}', ds_folder)
                
            if not match:
                match = re.search(r'run_[^_]+_(.*?)_\d{4}-\d{2}-\d{2}', ds_folder)
            
            if match:
                dataset_id = match.group(1)
                csv_path = os.path.join(run_path, ds_folder, "rounds.csv")
                
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        final_row = df[df['round'] == 51]
                        
                        if not final_row.empty:
                            all_observations.append({
                                "dataset_id": dataset_id,
                                "run_source": run_folder, 
                                "accuracy": final_row["Accuracy"].values[0] if "Accuracy" in final_row else final_row["accuracy"].values[0],
                                "balanced_accuracy": final_row["balanced_Accuracy"].values[0] if "balanced_Accuracy" in final_row else final_row["balanced_accuracy"].values[0],
                                "statistical_parity": final_row["Statistical_Parity"].values[0] if "Statistical_Parity" in final_row else final_row["statistical_parity"].values[0],
                                "equalized_odds": final_row["Equalized_Odds"].values[0] if "Equalized_Odds" in final_row else final_row["equalized_odds"].values[0]
                            })
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")

    full_df = pd.DataFrame(all_observations)

    final_averages = full_df.groupby("dataset_id").agg({
        'accuracy': 'mean',
        'balanced_accuracy': 'mean',
        'statistical_parity': 'mean',
        'equalized_odds': 'mean',
        'run_source': 'count' 
    }).rename(columns={'run_source': 'sample_size'}).reset_index()

    return final_averages


if __name__ == "__main__":
    ALGORITHM_NAME = "FedAvg"
    results = aggregate_benchmarks(f"logs/{ALGORITHM_NAME}")
    results.to_csv(f"./BenchmarkingEvaluations/final_dataset_averages_{ALGORITHM_NAME}.csv", index=False)
    print(results)