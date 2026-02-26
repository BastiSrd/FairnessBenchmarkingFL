from pathlib import Path
import csv


METRICS = ["accuracy", "balanced_accuracy", "statistical_parity", "equalized_odds"]
ROUND = "51"

# short prefixes for output headers (user requested acc1, acc2, acc3 style)
PREFIX = {
    "accuracy": "acc",
    "balanced_accuracy": "bal_acc",
    "statistical_parity": "stat_par",
    "equalized_odds": "eq_odds",
}

def parse_run_name(run_dir_name: str):
    # expected format: run_FedMinMax_{dataset}_{split}_timestamp
    parts = run_dir_name.split("_")
    if len(parts) < 4:
        return None, None
    dataset = parts[5]
    split = parts[6]
    return dataset, split


def extract_round_metrics(rounds_csv: Path, round_number: str):
    with rounds_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("round", "")).strip() == str(round_number):
                return {m: row.get(m, "") for m in METRICS}
    return None


def main(Algorithm: str):
    base = Path(__file__).parent.parent#
    logs_dir = base / "logs" / Algorithm
    run_roots = [logs_dir / f"{Algorithm}1", logs_dir / f"{Algorithm}2", logs_dir / f"{Algorithm}3"]

    # Collect data keyed by dataset_split
    data = {}

    # Order of runs (columns) will follow fed_roots order
    run_names = [p.name for p in run_roots]

    for root in run_roots:
        for run_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
            dataset, split = parse_run_name(run_dir.name)
            if dataset is None:
                continue
            key = f"{dataset}_{split}"
            if key not in data:
                data[key] = {"dataset": dataset, "split": split, "runs": {}}
            rounds_csv = run_dir / "rounds.csv"
            metrics = extract_round_metrics(rounds_csv, ROUND)
            data[key]["runs"].setdefault(root.name, metrics)

    # Prepare output CSV
    out_file = logs_dir / f"combinedFinalResults{Algorithm}.csv"
    # single combined dataset_split column
    header = ["dataset_split"]
    # for each metric, create columns like acc1, acc2, acc3 (order follows run_names)
    for m in METRICS:
        prefix = PREFIX.get(m, m)
        for idx in range(len(run_names)):
            header.append(f"{prefix}{idx+1}")

    with out_file.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for key in sorted(data.keys()):
            row = [f"{data[key]['dataset']}_{data[key]['split']}"]
            runs = data[key]["runs"]
            for m in METRICS:
                for rn in run_names:
                    metrics = runs.get(rn)
                    row.append(metrics.get(m))
            writer.writerow(row)

    print(f"Wrote combined CSV: {out_file}")


if __name__ == "__main__":
    main("GlobalGroupEodd")
