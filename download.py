import pandas as pd
from folktables import ACSDataSource, ACSIncome
import os

# Configuration
years = ['2018']
states = [
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga',
    'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md',
    'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
    'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc',
    'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'
]

# Create an output directory if it doesn't exist
output_dir = "./Datasets/acs_dataset"
os.makedirs(output_dir, exist_ok=True)
for state in states:
    state_frames = []
    print(f"Processing State: {state.upper()}")
    
    for year in years:
        try:
            # 1. Download data
            data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
            # folktables get_data expects uppercase state codes
            raw_data = data_source.get_data(states=[state.upper()], download=True)
            
            # 2. Transform using ACSIncome definition
            features, label, group = ACSIncome.df_to_numpy(raw_data)
            
            # 3. Create DataFrame
            df_year = pd.DataFrame(features, columns=ACSIncome.features)
            df_year['PINCP'] = label # Target variable named exactly for your function

            output_path = os.path.join(output_dir, f"{state}_data.csv")
            df_year.to_csv(output_path, index=False)
            print(f"  - {year} completed.")
            
        except Exception as e:
            print(f"  - Error for {state} in {year}: {e}")