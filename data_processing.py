#TODO removed CHEMBL1614414 bc 8000 molecules -- figure out how to subsample

import sys
import argparse
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, Any, Optional
from dpu_utils.utils import run_and_debug, RichPath

# Silence pandas warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('../ersilia-fsmol/fs_mol/preprocessing')
sys.path.append('../ersilia-fsmol/fs_mol/preprocessing/utils')
sys.path.append('../ersilia-fsmol/fs_mol/preprocessing/featurisers')

from clean import standardize
from featurise_utils import compute_sa_score
from molgraph_utils import molecule_to_graph
from save_utils import write_jsonl_gz_data
from featurize import filter_assays
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

def load_params():
    parser = argparse.ArgumentParser(description='Filtering bioassay arguments.')
    parser.add_argument('--save', type=bool, default=False,
                        help = "Whether or not we want to overwrite the existing csv files")
    parser.add_argument('--test_run', type=bool, default=False, 
                        help = "If test, we run the process_data function on a much smaller dataset")
    parser.add_argument('--num_processes', type=int, default=1,
                        help = "The number of processes to use when running FS-Mol functions")
    parser.add_argument('--load_metadata', type=str, default='../ersilia-fsmol/fs_mol/preprocessing/utils/helper_files',
                        help = "The path to the metadata directory for molecule graph featurization")
    parser.add_argument('--min_size_list', type=json.loads, default=[16],
                        help = """
                                A list minimum size of the assay (in terms of number of molecules tested). 
                                The code generates a dataset for each minimum size in the list.
                               """)
    parser.add_argument('--max_size', type=int, default=None,
                        help = "The maximum size of the assay (in terms of number of molecules tested)")
    parser.add_argument('--balance_limits', type=tuple, default=(10.0, 90.0),
                        help = "The lower and upper bound for the percentage of the molecules that must be active for the given assay.")
    parser.add_argument('--max_mol_weight', type=float, default=900.0,
                        help = "The molecular weight cutoff when cleaning each assay.")
    parser.add_argument('--sapiens_only', type=bool, default=False,
                        help = "SHOULD NOT BE CHANGED. Used in a call to an FS-Mol function.")
    parser.add_argument('--test_size_absolute', type=int, default=200,
                        help = "The size of the test set in terms of absolute assays.")
    parser.add_argument('--val_percentage', type=float, default=0.05,
                        help = "Percentage of non-test assays that should be used as validation.")
    parser.add_argument('--load_from_csv', type=bool, default=False,
                        help="True if we want to load from the standardized_df already saved rather than restandardize all molecules.")
    params = parser.parse_args()
    return params

def threshold_helper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply autothesholding procedure to each assay:

    1) Find the median for an assay
    2) Use the median as a threshold if 5 <= median(pchembl) <= 6. 
    If below round up to 5, if above round down to 6.
    3) Apply the threshold to the data series.

    For activity measurements, pchembl value is used.
        *This is opposed to FS-Mol which uses log standard value*

    Arguments:
        df: pd.DataFrame containing the data for a single assay

    Output:
        df: pd.DataFrame containing the data for a single assay with the active/inactive classification and threshold value

    ***Modified from autothreshold in ersilia-fsmol/fs_mol/preprocessing/utils/clean_utils.py.
    """

    # Remove index if it was saved with this file (back compatible)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Since we are grouping by chembl_id, we should drop the column (for recombining purposes)
    if 'chembl_id' in df.columns:
        df.drop(columns=['chembl_id'], inplace=True)


    # Set threshold limits
    threshold_limits = (5, 6)
    # Get median
    median = df["pchembl_value"].median()
    threshold = median

    # Round median to closest acceptable value
    if median < threshold_limits[0]:
        threshold = 5.0
    elif median > threshold_limits[1]:
        threshold = 6.0

    # Create active/inactive classification
    activity_benchmark = lambda x: True if x > threshold else False
    df['active'] = df['pchembl_value'].apply(activity_benchmark)
    # And save assay threshold
    df['assay_threshold'] = threshold

    return df

def implement_threshold(df: pd.DataFrame, params: argparse.Namespace) -> pd.DataFrame:
    """
        Given a dataframe read from Chembl, we 
            1. Filter out assays with fewer than min_size molecules
            2. Add an active/inactive classification for each molecule based on the given threshold
        Arguments:
            df: pd.DataFrame containing the data
            params: argparse.Namespace containing the arguments to the python file
        Output:
            filtered_df: pd.DataFrame containing the filtered data
        
    """
    # Group by assay
    gp = df.groupby('assay_id')
    x = gp.size()
    plt.hist(x, range = [0, 75], bins=15, weights=np.ones(len(x)) / len(x))
    plt.xlabel('Number of compounds')
    plt.ylabel('Percentage')
    plt.title('Compounds per assay')
    if params.save: 
        plt.savefig('compounds_per_assay.png')
    # Seems that ~60% of the assays have fewer than 5 compounds
    # We filter out compounds with fewer than min_size compounds

    # Grouping by assay
    # We make sure that each assay has more molecules than the minimum number in the min_size_lists
    # And also plus one because we need test set to be of at least size 2
    filtered_df = gp.filter(lambda x: len(x) > min(params.min_size_list) + 1)
    print('Post Initial Filtering, # of unique assays', len(filtered_df['assay_id'].unique()))
    print('Post Initial Filtering, # of unique compounds', len(filtered_df))

    # Plotting the histogram
    plt.clf()
    plt.hist(filtered_df['pchembl_value'])
    plt.xlabel('pchembl_value')
    plt.ylabel('Frequency')
    plt.title('Histogram of pchembl_value')
    if params.save: 
        plt.savefig("pchebml_value_histogram_min_size_{:02d}.png".format(min(params.min_size_list)))

    # Apply thresholding function
    filtered_df = filtered_df.groupby("chembl_id").apply(threshold_helper)
    # Print number of unique targets
    print("Number of unique targets:", filtered_df['target_pref_name'].nunique())
    print(f"Viruses: {filtered_df.loc[filtered_df['organism_taxonomy_l1'] == 'Viruses']['target_pref_name'].nunique()}, \
          Bacteria: {filtered_df.loc[filtered_df['organism_taxonomy_l1'] == 'Bacteria']['target_pref_name'].nunique()}, \
          Fungi: {filtered_df.loc[filtered_df['organism_taxonomy_l1'] == 'Fungi']['target_pref_name'].nunique()}"
          )

    # Save to new csv
    if params.save:
        filtered_df.to_csv('bioassay_table_filtered_active.csv', index=False)

    return filtered_df

def clean_assay(df: pd.DataFrame, 
                assay: str, 
                csv_writer: csv.DictWriter, 
                params: argparse.Namespace
                ) -> pd.DataFrame:
    """
        Given a dataframe containing data for a single assay, we clean the data by
            1. Standardizing the smiles string
            2. Removing > 900 Da moleculare weight
            3. Get log standard values
            4. Remove duplicates with conflicting measurements
        We then write the summary of the assay to summary.csv

        Arguments:
            df: pd.DataFrame containing data for a single assay
            assay: str, the assay chembl id
            csv_writer: csv.DictWriter object to write to summary.csv
            params: argparse.Namespace containing the arguments of the python file
        Output:
            df: pd.DataFrame, the cleaned data

        FS-Mol functions:
            Standardize: 
                From ersilia-fsmol/fs_mol/preprocessing/clean.py
                Called on the dataframe and performs cleaning steps 1-4.

        ***Modified from clean_assay in ersilia-fsmol/fs_mol/preprocessing/clean.py.
    """
    # Remove index if it was saved with this file (back compatible)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Since we are grouping by chembl_id, we should drop the column (for recombining purposes)
    if 'chembl_id' in df.columns:
        df.drop(columns=['chembl_id'], inplace=True)

    # The original size of the assay
    original_size = len(df)

    failed = False
    try:
        print(f"Processing {assay}.")
        # df = select_assays(df, **DEFAULT_CLEANING)                 # ersilia-fsmol/fs_mol/preprocessing/clean.py
        df = standardize(df, max_mol_weight = params.max_mol_weight) # ersilia-fsmol/fs_mol/preprocessing/clean.py
        # df = apply_thresholds(df, **DEFAULT_CLEANING)              # ersilia-fsmol/fs_mol/preprocessing/clean.py
    except Exception as e:
        df = None
        print(f"Failed cleaning on {assay} : {e}")
        failed = True

    if df is None or len(df) == 0:
        print(f"Assay {assay} was empty post cleaning.")
        failed = True

    # Create assay dict
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
            "year": "NaN"
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
            "num_pos": df["active"].sum(),
            "percentage_pos": df["active"].sum() * 100 / len(df),
            "max_mol_weight": df.iloc[0]["max_molecular_weight"],
            "threshold": df.iloc[0]["assay_threshold"],
            "max_num_atoms": df.iloc[0]["max_num_atoms"],
            "confidence_score": df.iloc[0]["confidence_score"],
            "standard_units": df.iloc[0]["standard_units"],
            "year": df.iloc[0]["year"]
        }

    # Write assay_dict as a row in csv
    csv_writer.writerow(assay_dict)
    return df

def split_assays(sdf, params) -> tuple:
    """
        Given a dataframe containing the data, we split the data into train, validation, and test sets.

        Arguments:
            df: pd.DataFrame, the data
            params: argparse.Namespace, the arguments to the python file
        Output:
            train_assays: list, the list of assays for training
            val_assays: list, the list of assays for validation
            test_assays: list, the list of assays for testing
    """
    # Keep only organisms that have been seen twice
    repeat_targets = sdf.groupby('target_id').filter(lambda x: len(x) > 1)
    unique_targets = sdf.groupby('target_id').filter(lambda x: len(x) == 1)

    # Dataframe is already sorted, keep first (newest) instance of each duplicated target and add to test list
    test_repeat_df = repeat_targets.loc[repeat_targets.sort_values('year', ascending=False).duplicated(subset='target_id', keep='first') == False]
    # Drop test assays from dataframe
    sdf.drop(test_repeat_df.index, inplace=True)
    # And store list of test assays. Given current dataset this is 75 assays
    test_list = test_repeat_df["chembl_id"].tolist()

    if len(test_list) < params.test_size_absolute:
        # If test set is not big enough. Add the most recent half of the unique assays.
        print("After using last assay for each duplicate target, test set not big enough")
        test_unique_df = unique_targets.sort_values('year', ascending=False).head(len(unique_targets) // 2)
        sdf.drop(test_unique_df.index, inplace=True)
        test_list += test_unique_df["chembl_id"].tolist()

        if len(test_list) < params.test_size_absolute:
            # If after adding half of unique tests, still not big enough, then add most recent assays until fill
            print("After using half of unique targets, test set not big enough")
            test_recent_df = sdf.sort_values('year', ascending=False).head(params.test_size_absolute - len(test_list))
            sdf.drop(test_recent_df.index, inplace=True)
            test_list += test_recent_df["chembl_id"].tolist()

    # Split remaining assays into train and validation
    sdf.sort_values('year', ascending=False, inplace = True)
    val_df, train_df = sdf.iloc[:int(len(sdf) * params.val_percentage)], sdf.iloc[int(len(sdf) * params.val_percentage):]
    val_list, train_list = val_df["chembl_id"].tolist(), train_df["chembl_id"].tolist()
    assert(set(val_list).isdisjoint(set(train_list)))
    assert(set(val_list).isdisjoint(set(test_list)))
    assert(set(train_list).isdisjoint(set(test_list)))
    return train_list, val_list, test_list 

def smiles_to_rdkit_mol(
    datapoint,
    include_fingerprints: bool = True,
    include_descriptors: bool = True,
    include_molecule_stats: bool = False,
    report_fail_as_none: bool = False,
) -> Optional[Dict[str, Any]]:
    """
        Given a datapoint (molecule) with a SMILES string, featurize it as an rdkit_molecule.

        Arguments:
            datapoint: Dict[str, Any], the molecule data
            include_fingerprints: bool, whether or not to include fingerprints
            include_descriptors: bool, whether or not to include descriptors
            include_molecule_stats: bool, whether or not to include molecule stats
            report_fail_as_none: bool, whether or not to report a failed molecule as None

        Output:
            Dict[str, Any], the processed molecule data with the rdkit_molecule added (or None)

        FS-Mol functions:
            compute_sa_score:
                From ersilia-fsmol/fs_mol/preprocessing/featurise_utils.py
                Computes the synthetic accessibility score of a molecule

        ***Adapted from smiles_to_rdkit_mol in ersilia-fsmol/fs_mol/preprocessing/featurise_utils.py
    """
    try:
        # Use canonical smiles as opposed to smiles as done in FS-Mol
        smiles_string = datapoint["canonical_smiles"]
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
                    datapoint["failed_to_convert_from_smiles"] = datapoint["canonical_smiles"]

        # Compute molecule-based scores with RDKit:
        if include_molecule_stats:
            datapoint["properties"] = {
                "sa_score": compute_sa_score(datapoint["mol"]), # ersilia-fsmol/fs_mol/preprocessing/featurise_utils.py
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

        3. Featurization
            3.1 Use rdkit to obtain molecule objects from the standardized smiles
            3.2 Use metadata to obtain graph representations of the molecules

        4. Save the data in FS-Mol format
            For each assay, we have a list of dictionaries where each dictionary represents a molecule post-cleaning and featurization
            For each assay, save this list of dictionaries in a jsonl.gz file with name the chembl id of the assay.

        THE FOLLOWING FUNCTION PERFORMS STEPS 1.2, 3, and 4.

        Arguments:
            df: pd.DataFrame, the filtered data
            params: argparse.Namespace, the arguments to the python file

        Output:
            None
            
        FS-Mol functions:
            filter_assays:
                From ersilia-fsmol/fs_mol/preprocessing/feauturize.py
                Uses the summary file to filter out assays that 
                    1. Are not within min_size and max_size post cleaning
                    2. Do not have a percentage of active molecules within balance_limits

            molecule_to_graph:
                From ersilia-fsmol/fs_mol/preprocessing/molgraph_utils.py
                Converts a molecule to a graph representation using the metadata
                
            write_jsonl_gz_data:
                From ersilia-fsmol/fs_mol/preprocessing/utils/save_utils.py
                Writes the 'datapoints' of an assay; i.e. the list of dictionaries representing the molecules to a jsonl.gz file
    """
    if not params.load_from_csv:
        # Group dataframe by assay_id
        gb = df.groupby('chembl_id')
        if params.test_run:
            df.sort_values('chembl_id', inplace=True)
            gb = df.iloc[:500].groupby('chembl_id')

        assert(False)
        # Initialize a summary.csv
        csv_file = open("summary.csv", 'w', newline="")
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
            "year",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        # Apply clean to each assay
        standardize_df = gb.apply(lambda x: clean_assay(x, x.name, csv_writer, params))
        # Save standardized df
        standardize_df.to_csv('standardized_df.csv')
        # Close the summary.csv file
        csv_file.close()
    else:
        standardize_df = pd.read_csv('standardized_df.csv')

    for min_size in params.min_size_list:
        # For every min size in our list, we repeat the saving process
        # Inefficient since we are re-storing the same assay multiple times, but I think the computationally expensive part is the standardize
        # Which we don't re-do for all min sizes
        params.min_size = min_size
        # Refilter assays after cleaning
        sdf = filter_assays('summary.csv', params) # ersilia-fsmol/fs_mol/preprocessing/feauturize.py
        print("After additional cleaning, we have", len(sdf), "unique assays.")

        # At this point, all assays are in the assays_to_process list.
        train_assays, val_assays, test_assays = split_assays(sdf, params) 

        # Load metadata for featurizations
        if params.load_metadata:

            print(f"Loading metadata from dir {params.load_metadata}")
            metapath = RichPath.create(params.load_metadata)
            path = metapath.join("metadata.pkl.gz")
            metadata = path.read_by_file_suffix()
            atom_feature_extractors = metadata["feature_extractors"]

        else:
            raise ValueError(
                "Metadata must be loaded for this processing, please supply "
                "directory containing metadata.pkl.gz."
            )

        # Featurize and save data
        # Loop taken from run() of featurize.py
        stand_gb = standardize_df.groupby('chembl_id')
        for assay_list, folder_name in zip([train_assays, val_assays, test_assays], ['train', 'valid', 'test']):
            for assay in assay_list:
                # For each assay, get all the molecules
                group = stand_gb.get_group(assay)
                # Transform into a dictionary with keys the columns
                datapoints = group.to_dict('list')

                # We save the featurized molecules as a list of dictionaries
                feat_data_list = []
                for i in range(len(datapoints['smiles'])):
                    # Isolate the ith molecule
                    datapoint = {key: value[i] for key, value in datapoints.items()}
                    # Add rdkit_molecule information
                    featurized_datapoint = smiles_to_rdkit_mol(datapoint)
                    # And also represent it as a graph using the metadata
                    datapoint["graph"] = molecule_to_graph(datapoint["mol"], atom_feature_extractors) # ersilia-fsmol/fs_mol/preprocessing/featurisers/molgraph_utils.py
                    # Append to it to our list of featurized datapoints
                    feat_data_list.append(featurized_datapoint)
                
                # Store all the information for the assay as is done by FS-Mol code
                write_jsonl_gz_data(f"dataset/min_size_{min_size:02d}/{folder_name}/{assay}.jsonl.gz", feat_data_list, len_data=len(feat_data_list)) # ersilia-fsmol/fs_mol/preprocessing/utils/save_utils.py

if __name__ ==  '__main__':
    # Load parameters
    params = load_params()

    # Assuming you have a DataFrame named 'df' with the data and 'pchembl_value' as the column
    df = pd.read_csv('bioassay_table_filtered.csv')
    # Implement initial threshold and get descriptive statistics
    filtered_df = implement_threshold(df, params)
    # Prepare data into FS-Mol format
    # prepare_data(filtered_df, params)