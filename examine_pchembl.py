import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

# Assuming you have a DataFrame named 'df' with the data and 'pchembl_value' as the column
df = pd.read_csv('bioassay_table_filtered.csv')

# Group by assay
gp = df.groupby('assay_id')
x = gp.size()
plt.hist(x, range = [0, 75], bins=15, weights=np.ones(len(x)) / len(x))
plt.xlabel('Number of compounds')
plt.ylabel('Percentage')
plt.title('Compounds per assay')
plt.show()
# Seems that ~60% of the assays have fewer than 5 compounds
# We filter those out

# Grouping by assay
filtered_df = gp.filter(lambda x: len(x) > 5)
print('Post Filtering, # of unique assays', len(filtered_df['assay_id'].unique()))
print('Post Filtering, # of unique compounds', len(filtered_df))

# # Plotting the histogram
plt.hist(filtered_df['pchembl_value'])
plt.xlabel('pchembl_value')
plt.ylabel('Frequency')
plt.title('Histogram of pchembl_value')
plt.show()

filtered_df.to_csv('bioassay_table_filtered_active.csv')