import csv 
import pandas as pd

# Read in bioassay_table_filtered.csv
df = pd.read_csv('bioassay_table_filtered.csv')

# For each line of bioassay_table_filtered.csv, 
# if pchembl_value > 6 then active = true, otherwise active = false
df['active'] = df['pchembl_value'].apply(lambda x: 'true' if x > 6 else 'false')

# Save to new csv
df.to_csv('bioassay_table_filtered_active.csv', index=False)