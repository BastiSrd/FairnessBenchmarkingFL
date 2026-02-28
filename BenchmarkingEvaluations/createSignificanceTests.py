import pandas as pd
from scipy.stats import ttest_rel
import numpy as np

ALGORITHM1 = "FedAvg"
ALGORITHM2 = "GlobalGroup"

# Load the data
df_1 = pd.read_csv(f'./logs/{ALGORITHM1}/combinedFinalResults{ALGORITHM1}.csv')
df_2 = pd.read_csv(f'./logs/{ALGORITHM2}/combinedFinalResults{ALGORITHM2}.csv')

metrics = ['acc', 'bal_acc', 'stat_par', 'eq_odds']
results_ttest = []

# Per-split T-tests (n=3 seeds per split)
for index, row_Alg1 in df_1.iterrows():
    split_name = row_Alg1['dataset_split']
    # Match the row
    row_Alg2 = df_2[df_2['dataset_split'] == split_name].iloc[0]
    
    ttestValues = {'dataset_split': split_name}
    
    for metric in metrics:
        a = [row_Alg2[f'{metric}1'], row_Alg2[f'{metric}2'], row_Alg2[f'{metric}3']]
        b = [row_Alg1[f'{metric}1'], row_Alg1[f'{metric}2'], row_Alg1[f'{metric}3']]
        
        try:
            # Paired T-test on the 3 seeds
            t, p = ttest_rel(a, b)
        except Exception:
            p = np.nan
            t = np.nan
        
        ttestValues[f'p_{metric}'] = p
        ttestValues[f't_{metric}'] = t

    results_ttest.append(ttestValues)
    

ttest_df = pd.DataFrame(results_ttest)


print("\nFirst 5 rows of per-split P-values:")
print(ttest_df.head())

ttest_df.to_csv(f'BenchmarkingEvaluations/significance_ttest_{ALGORITHM1}_vs_{ALGORITHM2}.csv', index=False)