# Synthetic Lethality Data and Code

Datasets:

(https://www.dropbox.com/sh/yunbe8agnoetwuz/AAAFIvYZipsWaupSogsX8Qija?dl=0)
1. `cell_embedding.hkl`: Expression and mutation features for each cell from [DepMap ](https://depmap.org/portal/download/all/) 22Q4's `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `OmicsSomaticMutationsMatrixHotspot.csv`, and `OmicsSomaticMutationsMatrixDamaging.csv` datasets. It is processed so expression features are z-scored and the features for each cell are l2-normalized to 1.
2. `CRISPRGeneEffect_processed.hkl`: `CRISPRGeneEffect.csv` from [DepMap ](https://depmap.org/portal/download/all/) 22Q4, filtered for cells that we have mutation and expression features for.
3. `feature_importances.hkl`: Resulting feature importances from running this code.
4. `knockouts.hkl`: Ordered (random, but fixed) list of KOs.

Files:
- `train_all_knockouts.py`: Train one kernel regression model on all CRISPR data and get feature importances for each KO.
- `demo.py`: Use calculated feature importances to visualize feature importance distributions for a given KO.
- `EigenProPytorch/`: Batched https://github.com/EigenPro/EigenPro-pytorch implementation modified for this specific problem.

Feel free to direct any questions about the code to caic@mit.edu. 
