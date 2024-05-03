import os
import json
import sys

DATASET_PATH = sys.argv[1] # Load in dataset path name

file_names_dict = {}
for folder in ['train', 'valid', 'test']:
    file_names_dict[folder] = [name.split('.')[0] for name in os.listdir(DATASET_PATH + '/' + folder)]

with open(DATASET_PATH + '/entire_train_set.json', 'w') as f:
    json.dump(file_names_dict, f)
