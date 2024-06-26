# Few-shot learning for molecular drug discovery

## Project Outline
<p> Motivated by few-shot learning for drug discovery in low and middle income countries, we query the Chembl database to create a copmrehensive list of organism assays, evaluate existing few-shot learning methods from FS-Mol and Mhnfs on this new dataset.
</p>

## Code
`extract_table.py`: Performs the Chembl query and generates `bioassay_table_filtered.csv`.<br>
`data_processing.py`: Saves the assay information queried from Chembl in FS-Mol format.<br>
`create_dataset_list.py`: Simple function for creating a list of all CHEMBL data in train, valid, and test.<br>
`utils/*`: Debugging and dataset exmaination methods

## Data

`bioassay_table.csv`: First attempt at loading CHEMBL data.<br>
`bioassay_table_filtered.csv`: Loading organism CHEMBL data filtered based on criteria specified in `extract_table.py`.<br>
`bioassay_table_filtered_active.csv`: Dataframe, saved after *implement thresholds* method in `filter_table.py`.<br>
`dataset/min_size_04`: Contains the D_4 datast

## Supercloud Code Differences

All code should be the same except notebook.ipynb which is optimized for the supercloud environment on supercloud, and for Andrew's environment on the local ersilia-fsmol directory. Results are most up to date on Supercloud. 
