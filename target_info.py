import sys
import argparse
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def load_params():
    parser = argparse.ArgumentParser(description='Filtering bioassay arguments.')
    # parser.add_argument('--save', type=bool, default=False,
    #                     help = "Whether or not we want to overwrite the existing csv files")
    params = parser.parse_args()
    return params

def load_target_info():
    assay_summary = pd.read_csv('summary.csv')
    chembl_data   = pd.read_csv('standardized_df.csv')
    for size in [4, 8, 16]:
        path = f"dataset/min_size_{size:02d}"
        with open(f"{path}/entire_train_set.json") as f:
            entire_train_set = json.load(f)
        test_set_molecules = chembl_data.loc[chembl_data["chembl_id"].isin(entire_train_set['test'])]
        test_set_targets = test_set_molecules.drop_duplicates(subset='chembl_id')[['chembl_id', 
                                                                                  'assay_id', 
                                                                                  'target_type', 
                                                                                  'target_pref_name', 
                                                                                  'target_id', 
                                                                                  'assay_organism',
                                                                                  'assay_type',
                                                                                  'confidence_score', 
                                                                                  'organism_taxonomy_l1', 
                                                                                  'organism_taxonomy_l2',
                                                                                  'organism_taxonomy_l3']]
        test_set_targets = test_set_targets.merge(assay_summary.loc[assay_summary["chembl_id"].isin(entire_train_set['test'])], validate='one_to_one')
        test_set_targets.to_csv(f"{path}/test_set_targets.csv", index=False)

if __name__ ==  '__main__':
    # Load parameters
    params = load_params()
    load_target_info()

