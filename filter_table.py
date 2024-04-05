import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../FS-Mol-Orgs/fs_mol/preprocessing')
sys.path.append('../FS-Mol-Orgs/fs_mol/preprocessing/featurisers')

from clean import *
from featurise_utils import *

ACTIVE_CUTOFF = 6
FILTER_CRITERIA = 16
SAVE = True

def filter_assays(df):
    # Group by assay
    gp = df.groupby('assay_id')
    x = gp.size()
    plt.hist(x, range = [0, 75], bins=15, weights=np.ones(len(x)) / len(x))
    plt.xlabel('Number of compounds')
    plt.ylabel('Percentage')
    plt.title('Compounds per assay')
    if SAVE: 
        plt.savefig('compounds_per_assay.png')
    # Seems that ~60% of the assays have fewer than 5 compounds
    # We filter out compounds with fewer than FILTER_CRITERIA compounds

    # Grouping by assay
    filtered_df = gp.filter(lambda x: len(x) > FILTER_CRITERIA)
    print('Post Filtering, # of unique assays', len(filtered_df['assay_id'].unique()))
    print('Post Filtering, # of unique compounds', len(filtered_df))

    # Plotting the histogram
    plt.clf()
    plt.hist(filtered_df['pchembl_value'])
    plt.xlabel('pchembl_value')
    plt.ylabel('Frequency')
    plt.title('Histogram of pchembl_value')
    if SAVE: 
        plt.savefig('pchebml_value_histogram.png')

    # For each line of bioassay_table_filtered.csv, 
    # if pchembl_value > cutoff then active = true, otherwise active = false
    activity_benchmark = lambda x: 'true' if x > ACTIVE_CUTOFF else 'false'
    filtered_df['active'] = filtered_df['pchembl_value'].apply(activity_benchmark)

    # Save to new csv
    if SAVE:
        filtered_df.to_csv('bioassay_table_filtered_active.csv', index=False)

    return filtered_df

def prepare_data(df):
    """
        We now want to store the data as it is stored by FSMol. 
        The ExtractDataset notebook and preprocessing folder contain useful information.

        After loading in the assays
            (information: chembl_id, assay_type, molregno_num, confidence_score),
        and for each assay relevant data
            (information: CHEMBL_ASSAY_PROTEIN query in preprocessing/utils/queries.py),

        FS-MOL performs the following steps:
        1. 'Clean' the data by 
            1.1 Removing all assays that do not have units %, uM, nM

            1.2 Standardizing
                1.2a Standardize the smiles string
                    note: I changed a function in cleaning_utils.py to deal with an error.
                1.2b Remove > 900 Da moleculare weight
                1.2c get log standard values
                1.2d Remove any repeats with conflicting measurements

            1.3 Applying a thresholding technique based on median activity measurement of the assay (given some criteria)
                1.3a The default is 5 if criteria are not met

        2. Classify proteins and split into tran/test/validation

        3. Featurize the SMILES string to created to create rdkit mol objects.

        THE FOLLOWING FUNCTION PERFORMS STEPS 1.2 and (TODO) 3
            the useful functions for this feauturization are in feautrisers subfolder of preprocessing
            I believe we only care about the _smiles_to_rdkit_mol function, 
            the functions above in the hierarchy are related to how they store their data.
    """
    # small_df = df.copy().head(100)

    standard_df = standardize(df)
    return standard_df

if __name__ ==  '__main__':
    # Assuming you have a DataFrame named 'df' with the data and 'pchembl_value' as the column
    df = pd.read_csv('bioassay_table_filtered.csv')
    filtered_df = filter_assays(df)
    standard_df = prepare_data(filtered_df)
    standard_df.to_csv('bioassay_table_standard.csv')