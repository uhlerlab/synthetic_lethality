# Synthetic Lethality Data and Code

Datasets:
1. `final_X_tcga_processed.hkl`: Expression and mutation features for each cell from [DepMap ](https://depmap.org/portal/download/all/) 22Q4's `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `OmicsSomaticMutationsMatrixHotspot.csv`, and `OmicsSomaticMutationsMatrixDamaging.csv` datasets. It is processed so expression features are z-scored and the features for each cell are l2-normalized to 1.
2. `final_X_tcga_raw_unnormalized.hkl`: Expression and mutation features for each cell from [DepMap ](https://depmap.org/portal/download/all/) 22Q4's `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `OmicsSomaticMutationsMatrixHotspot.csv`, and `OmicsSomaticMutationsMatrixDamaging.csv` datasets.
3. `CRISPRGeneEffect_processed.hkl`: `CRISPRGeneEffect.csv` from [DepMap ](https://depmap.org/portal/download/all/) 22Q4, filtered for cells that we have mutation and expression features for.
4. `Chronos_Combined_predictability_results.csv`: Predictability data from [DepMap ](https://depmap.org/portal/download/all/)
5. `cancerGeneList.tsv`: OncoKB cancer genes (https://www.oncokb.org/cancer-genes)
6. `sample_info.csv`: [DepMap ](https://depmap.org/portal/download/all/) metadata for cell lines
7. `datasets/tcga_data_processed_figures.hkl`: TCGA data downloaded from [Xena ]([https://depmap.org/portal/download/all/](https://xenabrowser.net/datapages/?host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443))

Files:
- `train_and_get_grads.ipynb`: Train one kernel regression model per knockout and get feature importances for each KO.
- `demo.py`: Use calculated feature importances to visualize feature importance distributions for a given KO.
- `generate_figures.ipynb`: Generate main text figures

Feel free to direct any questions about the code to caic@mit.edu. 
