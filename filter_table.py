import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

ACTIVE_CUTOFF = 6

# Assuming you have a DataFrame named 'df' with the data and 'pchembl_value' as the column
df = pd.read_csv('bioassay_table_filtered.csv')

# Group by assay
gp = df.groupby('assay_id')
x = gp.size()
plt.hist(x, range = [0, 75], bins=15, weights=np.ones(len(x)) / len(x))
plt.xlabel('Number of compounds')
plt.ylabel('Percentage')
plt.title('Compounds per assay')
plt.savefig('compounds_per_assay.png')
# Seems that ~60% of the assays have fewer than 5 compounds
# We filter those out

# Grouping by assay
filtered_df = gp.filter(lambda x: len(x) > 5)
print('Post Filtering, # of unique assays', len(filtered_df['assay_id'].unique()))
print('Post Filtering, # of unique compounds', len(filtered_df))

# # Plotting the histogram
plt.clf()
plt.hist(filtered_df['pchembl_value'])
plt.xlabel('pchembl_value')
plt.ylabel('Frequency')
plt.title('Histogram of pchembl_value')
plt.savefig('pchebml_value_histogram.png')

# For each line of bioassay_table_filtered.csv, 
# if pchembl_value > cutoff then active = true, otherwise active = false
activity_benchmark = lambda x: 'true' if x > ACTIVE_CUTOFF else 'false'
filtered_df['active'] = filtered_df['pchembl_value'].apply(activity_benchmark)

# Save to new csv
filtered_df.to_csv('bioassay_table_filtered_active.csv', index=False)