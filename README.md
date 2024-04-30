# Few-shot learning for molecular drug discovery

## Project Outline
<p> Motivated by few-shot learning for drug discovery in low and middle income countries, we query the Chembl database to create a copmrehensive list of organism assays, evaluate existing few-shot learning methods from FS-Mol and Mhnfs on this new dataset, and fine-tune the most performant model specifically for the organism assays.
</p>

## Code
`extract_table.py`: Performs the Chembl query and generates `bioassay_table_filtered.csv`.<br>
`data_processing.py`: Saves the assay information queried from Chembl in FS-Mol format.<br>
`utils/*`: Debugging and dataset exmaination methods

## Data

`bioassay_table.csv`: First attempt at loading CHEMBL data.<br>
`bioassay_table_filtered.csv`: Loading organism CHEMBL data filtered based on criteria specified in `extract_table.py`.<br>
`bioassay_table_filtered_active.csv`: Dataframe, saved after *implement thresholds* method in `filter_table.py`.<br>
`dataset/test/*`: Contains all test assays in FS-Mol format. <br>
`dataset/valid/*`: Contains all validation assays in FS-Mol format. <br>
`dataset/train/*`: Contains all train assay in FS-Mol format. <br>

## Instructions for Supercloud

Use anaconda/2023a for protonet

For multitask, use anaconda.2023a-pytorch and request a GPU: LLsub -i -s 20 -g volta:1