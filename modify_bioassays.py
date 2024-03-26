import csv 
import pandas as pd
ACTIVE_CUTOFF = 6


# Read in bioassay_table_filtered.csv
df = pd.read_csv('bioassay_table_filtered.csv')

# For each line of bioassay_table_filtered.csv, 
# if pchembl_value > cutoff then active = true, otherwise active = false
activity_benchmark = lambda x: 'true' if x > ACTIVE_CUTOFF else 'false'
df['active'] = df['pchembl_value'].apply(activity_benchmark)

# Save to new csv
df.to_csv('bioassay_table_filtered_active.csv', index=False)