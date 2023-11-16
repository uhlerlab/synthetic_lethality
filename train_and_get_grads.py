import sys

import hickle as hkl
import numpy as np
import torch
from scipy.stats import zscore

sys.path.insert(1, './recursive_feature_machines/')
from recursive_feature_machines import multi_rfm, rfm

def run_rfm():
    embedding = hkl.load("datasets/final_X_tcga_processed.hkl") # cell embedding in Dropbox
    gene_effects_df = hkl.load("datasets/CRISPRGeneEffect_processed.hkl") # viability scores in Dropbox
    bandwidth = 1
    reg = 1e-5
    num_iters = 1
    device = "cuda:1"
    
    features = embedding.columns
    embedding = embedding.fillna(0)

    X = embedding.values
    knockouts = gene_effects_df.columns

    # for each KO, train an RFM
    for knockout in knockouts:
        y = gene_effects_df[knockout].fillna(0).to_numpy().flatten().reshape((-1, 1)).astype("float64")
        model = rfm.RFM(device="cuda:3")
        model = model.fit(X,y, num_iters=num_iters, reg=reg, bandwidth=bandwidth,
            centering=False, verbose=False, diag_only=True)

        # get grads
        M = model.get_M()
    
        y = y.flatten()

        # weight each feature with its PCC
        for j in range(len(embedding.columns)):
            x = embedding.columns[j]
            if x.split("_")[-1] == "exp": # exp feature, only calculate the PCC of samples within 3 std
                ind = np.abs(zscore(embedding[x].values)) < 3
                r = np.corrcoef([embedding[x].values[ind], y[ind]])[0,1]
            else: # mut feature
                r = np.corrcoef([embedding[x].values, y])[0,1]
                # filter out mut features with positive correlation
                r = min(r,0)
            M[j] *= abs(r)
        
        data = np.concatenate((features.to_numpy().reshape((-1,1)),M.reshape((-1,1))),axis=1)
        np.savetxt("results/{}.csv".format(knockout),data,delimiter = ",",fmt='%s')
        print(knockout)

run_rfm()