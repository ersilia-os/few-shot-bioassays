import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../FS-Mol-Orgs/fs_mol/preprocessing')
sys.path.append('../FS-Mol-Orgs/fs_mol/preprocessing/utils')
from clean import *
from featurize import *

from standardizer import Standardizer
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

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
        they 'clean' the data by 
            1. Removing all assays that do not have units %, uM, nM
            2. Standardizing SMILES
            3. Applying a thresholding technique based on median activity measurement of the assay (given some criteria)
                3a. The default is 5 if criteria are not met
        they split into train/test/validation in part based on a classification of the assay.
        That is, depending on the type of protein target.
        Finally, they 'featureize' the SMILES string and use it to create rdkit mol objects.

        Of these steps, those that seem relevant are the standardization of SMILES and the featurization.
    """

    # Get small sample from df
    small_df = df.head(10) 

    sm = Standardizer(canon_taut=True)

    def standardize_smile(x: str):
        try:
            mol = Chem.MolFromSmiles(x)
            mol_weight = MolWt(mol)  # get molecular weight to do downstream filtering
            num_atoms = mol.GetNumAtoms()
            standardized_mol, _ = sm.standardize_mol(mol)
            return Chem.MolToSmiles(standardized_mol), mol_weight, num_atoms
        except Exception:
            # return a fail as None (downstream filtering)
            return None

    standard = small_df["smiles"].apply(lambda row: standardize_smile(row))
    small_df["canonical_smiles"] = standard.apply(lambda row: row[0])
    small_df["molecular_weight"] = standard.apply(lambda row: row[1])
    small_df["num_atoms"] = standard.apply(lambda row: row[2])

    # clean_df = standardize_smiles(df)
    # print(clean_df)
    return small_df    

if __name__ ==  '__main__':
    # Assuming you have a DataFrame named 'df' with the data and 'pchembl_value' as the column
    df = pd.read_csv('bioassay_table_filtered.csv')
    filtered_df = filter_assays(df)
    clean_df = prepare_data(filtered_df)
    print(clean_df)