# Federated Learning Fairness Benchmarking

A comprehensive benchmarking suite for evaluating fairness-aware Federated Learning algorithms across multiple datasets and fairness metrics.

## Overview

This project implements and benchmarks several Federated Learning algorithms with fairness constraints:

- **FedAvg**
- **FedMinMax**
- **EOFedMinMax**
- **TrustFed**
- **Global_Group**
- **Global_Group_Eodd**

The framework evaluates algorithms using both accuracy and fairness metrics:
- **Accuracy**: Overall and balanced accuracy
- **Statistical Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal TPR and FPR across groups

## Installation

### Prerequisites

- Python 3.11+ (tested on 3.13.3)
- pip or conda

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux/Mac**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Algorithm Run

To train algorithms on datasets:

```bash
python main.py
```

```python
if __name__ == "__main__":
    LOADERS = [
        "adult_iid5", "adult_iid10", "adult_age3", "adult_age5", 
        "bank_iid5", "bank_iid10", "bank_age3", "bank_age5", 
        "kdd_iid5", "kdd_iid10", "kdd_age3", "kdd_age5", 
        "acs_iid5", "acs_iid10", "acs_state3", "acs_state5", 
        "cac_iid5", "cac_iid10", "cac_state3", "cac_state5"
    ]
    ALGORITHMS = ['FedAvg', 'FedMinMax', "EOFedMinMax", 'TrustFed']
    TRUSTFED_FAIRNESS = ["SP", "EO"]  # Statistical Parity or Equalized Odds
    NUM_RUNS = 3
    
    for run_idx in range(NUM_RUNS):
        print(f"=== Starting Experimental Run {run_idx + 1}/{NUM_RUNS} ===")
        for algorithm in ALGORITHMS:
            if algorithm == "TrustFed":
                for loader in LOADERS:
                    for fairness in TRUSTFED_FAIRNESS:
                        runFLSimulation(loaderID=loader, algorithmID=algorithm, trustfedFairness=fairness)
            else:
                for loader in LOADERS:
                    runFLSimulation(loaderID=loader, algorithmID=algorithm)
```
Edit the LOADERS list, ALGORITHMS list or NUM_RUNS to limit the training process


### Running Global Group Algorithms

Global Group algorithms (centralized learning) can be run separately. \
Check the run.md in the respective Global_Group or Global_Group_Eodd folders.

# Running the  Global Group full Benchmark
To automatically run the full federated benchmarking pipeline across multiple datasets and distribution splits, execute the benchmark script:

python run_benchmark_Global_Group.py

By default, the script will execute 3 runs across all 20 dataset configurations. To change this edit the variables at the top of the run_benchmark_Global_Group.py file:
```python
# Change this to limit the number of runs per dataset
RUNS = 3

# Remove specific splits from this list to run a smaller benchmark
LOADERS = [
    "adult_iid5", "adult_iid10", "adult_age3", "adult_age5", 
    "bank_iid5", "bank_iid10", "bank_age3", "bank_age5", 
    "kdd_iid5", "kdd_iid10", "kdd_age3", "kdd_age5", 
    "acs_iid5", "acs_iid10", "acs_state3", "acs_state5", 
    "cac_iid5", "cac_iid10", "cac_state3", "cac_state5"
]
```

### Batch Size Configuration

Different algorithms use different batch sizes:

- **FedAvg**: 128 (standard SGD) - `FedAvg/FedAvgClient.py`
- **FedMinMax**: Full batch - `OrigFedMinMax/OrigFedMinMaxClient.py`
- **EOFedMinMax**: Full batch (required for group-wise metrics) - `EOFedMinMax/EOFedMinMaxClient.py`
- **TrustFed**: 128 (standard SGD) - `TrustFed/TrustFedClient.py`

To change batch sizes, edit the respective client files.


### Analysis Scripts

Post-training analysis available in `BenchmarkingEvaluations/`:

```bash
# Average results across multiple runs 
python BenchmarkingEvaluations/createAveragedResults.py

# Combine final results for one algorithm differentiated by run
python BenchmarkingEvaluations/createCombinedFinalResults.py

# Perform statistical significance tests
python BenchmarkingEvaluations/createSignificanceTests.py

# Generate visualization plots
python BenchmarkingEvaluations/createTestRunPlots.py
```

