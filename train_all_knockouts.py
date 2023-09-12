import os
import warnings

import hickle as hkl
import numpy as np
import pandas as pd
import scipy.linalg
import torch
from numpy.linalg import norm, solve, svd

from EigenProPytorch import eigenpro, eigenpro_backup, kernel

# list of knockouts, random but fixed order
knockouts = hkl.load('knockouts.hkl')
gene_effects_df = hkl.load("datasets/CRISPRGeneEffect_processed.hkl")

# flatten the gene_effects_df matrix in column-major order
y = np.concatenate([gene_effects_df[ko].values for ko in knockouts],axis=0).astype('float32').reshape((-1,1))

def train_rfm():
    """
    Train one Laplacian kernel regression model on all cell + KO samples in the gene effects dataset.
    """
    cell_embedding = hkl.load("cell_embedding.hkl")
    cell_features = cell_embedding.columns
    cell_X = torch.tensor(cell_embedding.values)

    num_cell_features = len(cell_features)
    features_stripped = np.array([f.split("_")[0] for f in cell_features])
    num_knockouts = len(knockouts)

    device = "cuda:1" # or cpu
    y = torch.tensor(y).to(device)

    n = y.shape[0]
    # Each sample (cell + KO) is embedded with its cell features and a one-hot for knockout features
    d = num_cell_features + num_knockouts
    num_cells = n // num_knockouts

    cell_distances = kernel.euclidean_distances(cell_X,cell_X).to(device)
    kernel_fn = lambda x,y: kernel.laplacian(x, y, bandwidth=1)

    model = eigenpro.FKR_EigenPro(cell_distances,kernel_fn, cell_X, num_cells, num_knockouts, bandwidth, device=device)
    res = model.fit(y, epochs=range(5),mem_gb=900)
    hkl.dump(model.weight.reshape((1,-1)).detach().cpu().numpy(),"model_weights.hkl")
    return

def get_feature_importances():
    """
    Now that you've trained the model and saved the weights, get the feature importances for each KO.
    """
    cell_embedding = hkl.load("cell_embedding.hkl")
    cell_features = cell_embedding.columns
    features_stripped = np.array([f.split("_")[0] for f in features])

    device = "cuda:1"
    cell_X = torch.tensor(cell_embedding.values).to(device)
    num_cell_features = len(cell_features)
    cell_distances = kernel.euclidean_distances(cell_X,cell_X,squared=True).to(device)
    cell_distances = cell_distances.fill_diagonal_(0)
    num_knockouts = len(knockouts)

    n = y.shape[0]
    d = num_cell_features + num_knockouts
    num_cells = n // num_knockouts
    bandwidth = 1

    # Assumes that M = Id for RFM model
    M_init = torch.tensor(torch.ones(num_cell_features)).to(device)
    weight = torch.tensor(hkl.load("model_weights.hkl")).to(device)

    # Calculate a normalized Laplacian kernel for cell features from the distance matrix
    def K_from_dist(dist):
        dist = dist**0.5
        dist.clamp_(min=0)
        dist[dist < 1e-10] = 0
        K = torch.exp(-dist)
        with np.errstate(divide='ignore'):
            K = K/dist
        K[K == float("Inf")] = 0.
        return K.float()

    # Kernel for samples that are from the same KO
    dist_ko = kernel.euclidean_distances(cell_X,cell_X,squared=True).fill_diagonal_(0)
    K_ko = torch.nan_to_num(torch.exp(-dist_ko**0.5)).float()
    K_ko_normalized = K_from_dist(dist_ko)

    # Kernel for samples that are not from the same KO (all the distances increase by exactly 2 because the KO embeddings are one-hots)
    dist_all = dist_ko + 2
    K_all = torch.nan_to_num(torch.exp(-dist_all**0.5)).float()
    K_all_normalized = K_from_dist(dist_all)

    feature_importance_df = {}
    cell_X = cell_X.float()
    sol = weight.reshape((1,-1))
    y = y.reshape((num_knockouts,num_cells)).T
    for i in range(len(knockouts)):
        knockout = knockouts[i]

        # subset only to samples corresponding to knockout
        sub_y = y[:,i]
        # get the gradients of only the cell features of the specified samples
        sub_G = get_grads_sl(num_cells, num_knockouts, cell_X, cell_X, i, sol, bandwidth, M_init, K_ko_normalized, K_all_normalized)
        sub_G = np.nan_to_num(sub_G.detach().cpu().numpy())

        # the M vector (feature importances) is the average gradient over all samples for each feature
        M_i = np.mean(sub_G**2,axis=0).flatten()

        # weight the feature importances with the pcc of the feature and the viability scores
        for j in range(len(cell_embedding.columns)):
            x = cell_embedding.columns[j]

            # sometimes outlier viability scores skew the pcc, so we only calculate pcc for non-outlier values
            ind = np.abs(zscore(cell_embedding[x].values)) < 3
            r = np.corrcoef([cell_embedding[x].values[ind], sub_y.flatten()[ind]])[0,1]

            # want positive correlation for expression features and negative correlation for mutation features. 
            # expression features have _exp appended.
            if "_" in x:
                r = np.corrcoef([embedding[x].values[ind], sub_y.flatten()[ind]])[0,1]
            else:
                r = np.corrcoef([embedding[x].values, sub_y.flatten()])[0,1]
            if ("_" in x and r >= 0) or ("_" not in x and r <= 0):
                M_i[j] *= abs(r)
            else:
                M_i[j] *= 0

        feature_importance_df[knockout] = M_i.flatten()
    
    feature_importance_df = pd.DataFrame(feature_importance_df,columns=knockouts,index=features)
    hkl.dump(feature_importance_df,"feature_importances.hkl")

def get_grads_sl(num_cells,num_kos,cell_X, x, ko_id, sol, L, P, K_ko, K_all):
    """
    Get the cell feature gradients wrt the samples that correspond to knockout with index ko_id (i.e. knockouts[ko_id] = knockout)
    Note that the output shape is num_cells x num_cell_features.

    Assumes:
        1. centering = false
        2. diag_only = true and/or P is irrelevant

    @params
        - num_cells
        - num_kos
        - cell_X: cell embedding
        - x: what you are taking the gradient wrt. For our purposes, x = cell_X.
        - ko_id: ID of knockout you want the gradients for
        - sol: weights of trained model
        - L: bandwidth, assume L = 1
        - P: starting M (feature importances), assume M = Id
        - K_ko: normalized Laplacian kernel matrix for samples that are part of the same KO
        - K_all: normalized Laplacian kernel matrix for samples that are not part of the same KO
    """

    n,d = x.shape
    grad = np.zeros((n,d))

    # The below is just a batched implementation of:
    #   step2 = K(x,X) @ (weight * X * P)
    #   step3 = (weight @ K(X,x)).T * (x * P)
    #   return (step2 - step3) * -1/L

    sol = sol.T
    ko_sol = sol[num_cells*ko_id:num_cells*(ko_id+1)].flatten()
    notko_sol = torch.cat((sol[:num_cells*ko_id],sol[num_cells*(ko_id+1):]),0).flatten()
    notko_sol_shaped = notko_sol.reshape((num_kos-1,num_cells)).T
    notko_sol_shaped_sum = notko_sol_shaped.sum(axis=1)

    x = x * P
    step2 = K_all @ (x * notko_sol_shaped_sum.reshape((-1,1))) + K_ko @ (x * ko_sol.reshape((-1,1)))
    step3 = (notko_sol_shaped_sum.reshape((1,-1)) @ K_all + ko_sol.reshape((1,-1)) @ K_ko).T * x
    grad = (step2 - step3) * -1/L

    return grad

train_rfm()
get_feature_importances()