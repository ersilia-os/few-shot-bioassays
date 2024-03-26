import matplotlib.pyplot as plt
import csv
import pandas as pd

# Assuming you have a DataFrame named 'df' with the data and 'pchembl_value' as the column
df = pd.read_csv('bioassay_table_filtered.csv')

# Plotting the histogram
plt.hist(df['pchembl_value'])
plt.xlabel('pchembl_value')
plt.ylabel('Frequency')
plt.title('Histogram of pchembl_value')
plt.show()