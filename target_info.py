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
    chembl_data   = pd.read_csv('bioassay_table_filtered_active.csv')
    for size in [4, 8, 16]:
        path = f"dataset/min_size_{size:02d}"
        with open(f"{path}/entire_train_set.json") as f:
            entire_train_set = json.load(f)
        chembl_data.loc[chembl_data[].isin(entire_train_set['test'])]


if __name__ ==  '__main__':
    # Load parameters
    params = load_params()
    load_target_info()

