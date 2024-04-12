# Few-shot learning for molecular drug discovery

## Project Outline
<p> Motivated by few-shot learning for drug discovery in low and middle income countries, we use the Chembl database to create a useful set of organism assays, evaluate existing few-shot learning methods from FS-Mol and Mhnfs on this new dataset, before fine-tuning a model specifically for organism assays.
</p>

## Code
`extract_table.py`: Performs the Chembl query and generates `bioassay_table_filtered.csv`.<br>
`data_processing.py`: Saves the assay information queried form Chembl in FS-Mol.

## Data

`bioassay_table.csv`: First attempt at loading CHEMBL data.<br>
`bioassay_table_filtered.csv`: Loading organism CHEMBL data filtered based on criteria specified in `extract_table.py`.<br>
`bioassay_table_filtered_active.csv`: Final dataframe, saved after *implement thresholds* method in `filter_table.py`<br.>
