import subprocess
import sys
import os

print("=======================================================")
print("    Checking and Installing Requirements...")
print("=======================================================")

# Auto-Install Requirements
req_file = "requirements.txt"
if os.path.exists(req_file):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("Requirements installed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        print("Please check your internet connection or install manually.")
        sys.exit(1)
else:
    print(f"Warning: '{req_file}' not found. Skipping auto-install.\n")


print("=======================================================")
print("    Starting Global Group Fairness Federated Learning Benchmark")
print("=======================================================")

RUNS = 3
LOADERS = [
    "adult_iid5", "adult_iid10", "adult_age3", "adult_age5", 
    "bank_iid5", "bank_iid10", "bank_age3", "bank_age5", 
    "kdd_iid5", "kdd_iid10", "kdd_age3", "kdd_age5", 
    "acs_iid5", "acs_iid10", "acs_state3", "acs_state5", 
    "cac_iid5", "cac_iid10", "cac_state3", "cac_state5"
]

for loader in LOADERS:
    print(f"\n=======================================================")
    print(f"DATASET SPLIT: {loader}")
    print(f"=======================================================")
    
    for i in range(1, RUNS + 1):
        print(f"\n  --- Run {i} of {RUNS} for {loader} ---")
        
        print("  [1/2] Running Global_Group (Statistical Parity)...")
        subprocess.run([sys.executable, "-m", "Global_Group.main", "--loader", loader], check=True)
        
        print("  [2/2] Running Global_Group_Eodd (Equalized Odds)...")
        subprocess.run([sys.executable, "-m", "Global_Group_Eodd.main", "--loader", loader], check=True)

print("\n=======================================================")
print("                BENCHMARK COMPLETE!")
print("=======================================================")