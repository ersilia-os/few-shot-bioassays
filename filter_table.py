import sys
import logging
import argparse
import csv
import pandas as pd
import numpy as np
import functools
import matplotlib.pyplot as plt

sys.path.append('../FS-Mol-Orgs/fs_mol/preprocessing')
sys.path.append('../FS-Mol-Orgs/fs_mol/preprocessing/featurisers')
sys.path.append('../FS-Mol-Orgs/fs_mol/preprocessing/utils')

from clean import *
from featurise_utils import *
from save_utils import *
from rdkit import DataStructs
from rdkit.Chem import (
    Mol,
    RDConfig,
    Descriptors,
    MolFromSmiles,
    rdFingerprintGenerator,
)
from rdkit.Chem.QED import qed
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt, BertzCT

logger = logging.getLogger(__name__)

def load_params():
    parser = argparse.ArgumentParser(description='Filtering bioassay arguments.')
    parser.add_argument('--ACTIVE_CUTOFF', type=int, default=6)
    parser.add_argument('--FILTER_CRITERIA', type=int, default=16)
    parser.add_argument('--SAVE', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--num_processes', type=int, default=1)
    params = parser.parse_args()

    return params


def filter_assays(df, params):
    # Group by assay
    gp = df.groupby('assay_id')
    x = gp.size()
    plt.hist(x, range = [0, 75], bins=15, weights=np.ones(len(x)) / len(x))
    plt.xlabel('Number of compounds')
    plt.ylabel('Percentage')
    plt.title('Compounds per assay')
    if params.SAVE: 
        plt.savefig('compounds_per_assay.png')
    # Seems that ~60% of the assays have fewer than 5 compounds
    # We filter out compounds with fewer than FILTER_CRITERIA compounds

    # Grouping by assay
    filtered_df = gp.filter(lambda x: len(x) > params.FILTER_CRITERIA)
    logger.info('Post Filtering, # of unique assays', len(filtered_df['assay_id'].unique()))
    logger.info('Post Filtering, # of unique compounds', len(filtered_df))

    # Plotting the histogram
    plt.clf()
    plt.hist(filtered_df['pchembl_value'])
    plt.xlabel('pchembl_value')
    plt.ylabel('Frequency')
    plt.title('Histogram of pchembl_value')
    if params.SAVE: 
        plt.savefig('pchebml_value_histogram.png')

    # For each line of bioassay_table_filtered.csv, 
    # if pchembl_value > cutoff then active = true, otherwise active = false
    activity_benchmark = lambda x: 'true' if x > params.ACTIVE_CUTOFF else 'false'
    filtered_df['active'] = filtered_df['pchembl_value'].apply(activity_benchmark)

    # Save to new csv
    if params.SAVE:
        filtered_df.to_csv('bioassay_table_filtered_active.csv', index=False)

    return filtered_df

def clean_assay(df: pd.DataFrame, assay: str, csv_writer, params) -> pd.DataFrame:
    """
        Modified from clean_assay in FS-Mol_Orgs/fs_mol/preprocessing/clean.py
        Called on each assay. 
        Standardizes the smiles molecules for that assay, and writes a summary to csv_writer
    """
    # remove index if it was saved with this file (back compatible)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    # Since we are grouping by assay_id, we should drop this
    if 'assay_id' in df.columns:
        df.drop(columns=['assay_chembl_id'], inplace=True)

    original_size = len(df)

    failed = False
    try:
        logger.info(f"Processing {assay}.")
        # df = select_assays(df, **DEFAULT_CLEANING)
        df = standardize(df)
        # df = apply_thresholds(df, **DEFAULT_CLEANING)
    except Exception as e:
        df = None
        logger.warning(f"Failed cleaning on {assay} : {e}")
        failed = True

    if df is None or len(df) == 0:
        logger.warning(f"Assay {assay} was empty post cleaning.")
        failed = True
    
    assay_dict = {}
    if failed:
        assay_dict = {
            "chembl_id": assay,
            "target_id": "NaN",
            "assay_type": "NaN",
            "assay_organism": "NaN",
            "raw_size": "NaN",
            "cleaned_size": 0,
            "cleaning_failed": str(True),
            "cleaning_size_delta": "NaN",
            "num_pos": "NaN",
            "percentage_pos": "NaN",
            "max_mol_weight": "NaN",
            "threshold": "NaN",
            "max_num_atoms": "NaN",
            "confidence_score": "NaN",
            "standard_units": "NaN",
        }

    else:
        target_id = df.iloc[0]["target_id"] if "target_id" in df.columns else None

        organism = None if df.iloc[0]["assay_organism"] == "nan" else df.iloc[0]["assay_organism"]
        assay_dict = {
            "chembl_id": assay,
            "target_id": target_id,
            "assay_type": df.iloc[0]["assay_type"],
            "assay_organism": organism,
            "raw_size": original_size,
            "cleaned_size": len(df),
            "cleaning_failed": failed,
            "cleaning_size_delta": original_size - len(df),
            # "num_pos": df["activity"].sum(),
            # "percentage_pos": df["activity"].sum() * 100 / len(df),
            "max_mol_weight": df.iloc[0]["max_molecular_weight"],
            "threshold": params.ACTIVE_CUTOFF,
            "max_num_atoms": df.iloc[0]["max_num_atoms"],
            "confidence_score": df.iloc[0]["confidence_score"],
            "standard_units": df.iloc[0]["standard_units"],
        }

    csv_writer.writerow(assay_dict)
    return df

def smiles_to_rdkit_mol(
    datapoint,
    include_fingerprints: bool = True,
    include_descriptors: bool = True,
    include_molecule_stats: bool = False,
    report_fail_as_none: bool = False,
) -> Optional[Dict[str, Any]]:
    try:
        smiles_string = datapoint["smiles"]
        rdkit_mol = MolFromSmiles(smiles_string)

        datapoint["mol"] = rdkit_mol

        # Compute fingerprints:
        if include_fingerprints:
            datapoint["fingerprints_vect"] = rdFingerprintGenerator.GetCountFPs(
                [rdkit_mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
            fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
            DataStructs.ConvertToNumpyArray(datapoint["fingerprints_vect"], fp_numpy)
            datapoint["fingerprints"] = fp_numpy

        # Compute descriptors:
        if include_descriptors:
            datapoint["descriptors"] = []
            for descr in Descriptors._descList:
                _, descr_calc_fn = descr
                try:
                    datapoint["descriptors"].append(descr_calc_fn(rdkit_mol))
                except Exception:
                    datapoint["failed_to_convert_from_smiles"] = datapoint["SMILES"]

        # Compute molecule-based scores with RDKit:
        if include_molecule_stats:
            datapoint["properties"] = {
                "sa_score": compute_sa_score(datapoint["mol"]),
                "clogp": MolLogP(datapoint["mol"]),
                "mol_weight": ExactMolWt(datapoint["mol"]),
                "qed": qed(datapoint["mol"]),
                "bertz": BertzCT(datapoint["mol"]),
            }

        return datapoint
    except Exception:
        if report_fail_as_none:
            datapoint["mol"] = None
            return datapoint
        else:
            raise

def prepare_data(df, params):
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
    """
    # Group dataframe by assay_id
    gb = df.groupby('assay_chembl_id')
    if params.test:
        gb = df.iloc[:10].groupby('assay_chembl_id')

    # Initialize a summary.csv
    csv_file = open('summary.csv', 'w', newline="")
    fieldnames = [
        "chembl_id",
        "target_id",
        "assay_type",
        "assay_organism",
        "raw_size",
        "cleaned_size",
        "cleaning_failed",
        "cleaning_size_delta",
        "num_pos",
        "percentage_pos",
        "max_mol_weight",
        "threshold",
        "max_num_atoms",
        "confidence_score",
        "standard_units",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Apply clean to each assay
    standardize_df = gb.apply(lambda x: clean_assay(x, x.name, csv_writer, params))
    # Close the summary.csv file
    csv_file.close()

    # Featurize and save data
    # Adapted from run() in featurize.py
    assays = set()
    for name, group in standardize_df.groupby('assay_chembl_id'):
        assays.add(name)
        datapoints = group.to_dict('list')
        feat_data_list = []
        for i in range(len(datapoints['smiles'])):
            datapoint = {key: value[i] for key, value in datapoints.items()}
            featurized_datapoint = smiles_to_rdkit_mol(datapoint)
            feat_data_list.append(featurized_datapoint)

        featurized_datapoints = {}
        for d in feat_data_list:
            for k, v in d.items():
                featurized_datapoints[k] = featurized_datapoints.get(k, []) + [v]
        
        print(featurized_datapoints)
        # After this step, they save the data as a FeaturisedData object which 
        # automatically does graph featurization 

        # The final setp is to save as a jsonl.gz file
        write_jsonl_gz_data(f"data/{name}.jsonl.gz", featurized_datapoints, len_data=len(featurized_datapoints))

    return standard_df

if __name__ ==  '__main__':
    # Load parameters
    params = load_params()

    # Assuming you have a DataFrame named 'df' with the data and 'pchembl_value' as the column
    df = pd.read_csv('bioassay_table_filtered.csv')
    filtered_df = filter_assays(df, params)
    standard_df = prepare_data(filtered_df, params)
    standard_df.to_csv('bioassay_table_standard.csv')