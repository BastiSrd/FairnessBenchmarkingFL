import os
import json
import csv
from datetime import datetime

class FLLogger:
    """
    A logger for Federated Learning simulations that tracks per-round metrics,
    client reports, and best metrics. Logs are then saved to CSV, JSON, and a .log file.
    """

    def __init__(self, algorithm, loader, config):
        """
        Initializes the logger, creates directories and files.

        Args:
            algorithm (str): Name of the FL algorithm used (e.g., 'FedAvg').
            loader (str): Identifier of the data loading method (e.g., '3_clients').
            config (dict): Additional configuration parameters such as learning rate,
                           number of client epochs, total rounds, etc.

        Creates:
            - rounds.csv: stores per-round metrics
            - clients.csv: stores client-level metrics per round
            - run.log: log of the simulation
            - run_dir: directory for all log files
            - summary.json: containing configuration and best metrics for Accuracy, Statistical Parity, Equalized Odds
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = f"logs/run_{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)

        #Track best metrics
        self.best_metrics = {
            "Accuracy": {"round": -1, "value": -1},
            "Statistical_Parity": {"round": -1, "value": float('inf')},
            "Equalized_Odds": {"round": -1, "value": float('inf')}
        }

        #Save configuration
        self.config = {
            "algorithm": algorithm,
            "loader": loader,
            **config,
            "timestamp": timestamp
        }

        #CSV file paths
        self.rounds_path = os.path.join(self.run_dir, "rounds.csv")
        self.clients_path = os.path.join(self.run_dir, "clients.csv")
        self.log_file_path = os.path.join(self.run_dir, "run.log")

        #Initialize rounds CSV
        with open(self.rounds_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "accuracy", "statistical_parity", "equalized_odds"])

        #Initialize clients CSV
        with open(self.clients_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "client", "loss", "samples", "weight"])

        #Initialize .log file
        with open(self.log_file_path, "w") as f:
            f.write(f"FL Simulation Log - {timestamp}\n")
            f.write(f"Algorithm: {algorithm}, Loader: {loader}\n")
            f.write(f"Config: {json.dumps(config)}\n")
            f.write("="*50 + "\n")

    def log_round(self, round_idx, metrics):
        """
        Logs per-round metrics.

        Args:
            round_idx (int): Current round number.
            metrics (dict): Dictionary containing metrics for this round:
                - "Accuracy"
                - "Statistical_Parity"
                - "Equalized_Odds"

        Actions:
            - Adds metrics to rounds.csv
            - Updates best metrics if applicable
            - Adds a entry to run.log
        """
        #Write to CSV
        with open(self.rounds_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round_idx,
                metrics["Accuracy"],
                metrics["Statistical_Parity"],
                metrics["Equalized_Odds"]
            ])

        #Update best metrics
        if metrics["Accuracy"] > self.best_metrics["Accuracy"]["value"]:
            self.best_metrics["Accuracy"] = {"round": round_idx, "value": metrics["Accuracy"]}
        if abs(metrics["Statistical_Parity"]) < self.best_metrics["Statistical_Parity"]["value"]:
            self.best_metrics["Statistical_Parity"] = {"round": round_idx, "value": abs(metrics["Statistical_Parity"])}
        if metrics["Equalized_Odds"] < self.best_metrics["Equalized_Odds"]["value"]:
            self.best_metrics["Equalized_Odds"] = {"round": round_idx, "value": metrics["Equalized_Odds"]}

        #Add to .log file
        with open(self.log_file_path, "a") as f:
            f.write(f"Round {round_idx}: Accuracy={metrics['Accuracy']:.4f}, "
                    f"SP={metrics['Statistical_Parity']:.4f}, "
                    f"EO={metrics['Equalized_Odds']:.4f}\n")

    def log_clients(self, round_idx, client_reports):
        """
        Logs client-level data for a given round.

        Args:
            round_idx (int): Current round number.
            client_reports (list of dict): Each dict contains:
                - "client_name"
                - "loss"
                - "samples"
                - "weights" (state_dict)
        
        Actions:
            - Adds client info to clients.csv
            - Adds a summary to run.log
        """
        total_samples = sum(r["samples"] for r in client_reports)

        #Write to CSV
        with open(self.clients_path, "a", newline="") as f:
            writer = csv.writer(f)
            for r in client_reports:
                weight = r["samples"] / total_samples
                writer.writerow([
                    round_idx,
                    r["client_name"],
                    r["loss"],
                    r["samples"],
                    weight
                ])

        #Add to .log file
        with open(self.log_file_path, "a") as f:
            f.write(f"--- Client Reports Round {round_idx} ---\n")
            for r in client_reports:
                weight = r["samples"] / total_samples
                f.write(f"{r['client_name']}: Loss={r['loss']:.4f}, "
                        f"Samples={r['samples']}, Weight={weight:.4f}\n")
            f.write("\n")

    def finalize(self):
        """
        Writes the final summary of the simulation.

        Actions:
            - Saves a single summary.json containing:
                - Configuration
                - Best metrics for Accuracy, Statistical Parity, Equalized Odds
            - Writes best metrics to the run.log
            - Prints the log directory
        """
        #Write summary JSON
        full_summary = {
            "config": self.config,
            "best_metrics": self.best_metrics
        }

        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(full_summary, f, indent=4)

        #Adds best metrics to .log
        with open(self.log_file_path, "a") as f:
            f.write("\n--- Best Metrics ---\n")
            for metric_name, info in self.best_metrics.items():
                f.write(f"Best {metric_name}:\n")
                f.write(f"  Round: {info['round']}\n")
                f.write(f"  Value: {info['value']}\n\n")
        print(f"\nLogs successfully written to: {self.run_dir}")
