# Synthetic Lethality Data and Code

Datasets:

(https://www.dropbox.com/sh/yunbe8agnoetwuz/AAAFIvYZipsWaupSogsX8Qija?dl=0)
1. `final_X_tcga_processed.hkl`: Expression and mutation features for each cell from [DepMap ](https://depmap.org/portal/download/all/) 22Q4's `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `OmicsSomaticMutationsMatrixHotspot.csv`, and `OmicsSomaticMutationsMatrixDamaging.csv` datasets. It is processed so expression features are z-scored and the features for each cell are l2-normalized to 1.
2. `CRISPRGeneEffect_processed.hkl`: `CRISPRGeneEffect.csv` from [DepMap ](https://depmap.org/portal/download/all/) 22Q4, filtered for cells that we have mutation and expression features for.
3. `feature_importances.hkl`: Resulting feature importances from running this code.

Files:
- `train_and_get_grads.py`: Train one kernel regression model per knockout and get feature importances for each KO.
- `demo.py`: Use calculated feature importances to visualize feature importance distributions for a given KO.

Feel free to direct any questions about the code to caic@mit.edu. 
