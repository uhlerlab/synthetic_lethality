
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hickle as hkl
from scipy.stats import zscore

embedding = hkl.load("embeddings/cell_embedding.hkl")
gene_effects_df = hkl.load("datasets/2023/CRISPRGeneEffect_processed.hkl")
feature_importance_df = hkl.load("feature_importances.hkl")

# %% 
def get_top_indicators(M,k=10,feature_type="genes",importances=False):
    M = M.sort_values(ascending=False)
    if feature_type == "genes":
        features_sorted = [x.split("_")[0] for x in M.index]
        seen = set()
        genes_ordered = []
        for g in features_sorted:
            if g not in seen:
                genes_ordered.append(g)
                seen.add(g)
        return genes_ordered[:k]
    else:
        if importances:
            return M.loc[M.index[:k]].to_string(index=True,header=False) ## prints out actual feature importances
        else:
            return [x for x in M.index[:k] if x.split("_")[-1] != "exp"]

def plot_feature_importances(knockout,M):
    feature_importances = M.to_numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    n, bins, patches = ax.hist(feature_importances)
    ax.set_title("{}".format(knockout))
    ax.set_xlabel("Feature Importance (adjusted)")
    ax.set_ylabel("Frequency (log scale)")
    plt.yscale('log')
    plt.show()

def plot_pcc(knockout,M):
    M = M.sort_values(ascending=False)
    top_feature = M.index[0]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    exp_feature = top_feature.split("_")[-1] == "exp"
    ind = np.abs(zscore(embedding[top_feature].values)) < 3
    x = embedding[top_feature][ind]
    y = gene_effects_df[knockout][ind]
    ax.scatter(x,y,c="green" if exp_feature else "darkviolet")
    r = np.corrcoef([x,y])[0,1]
    if exp_feature:
        ax.set_xlabel("Gene expression TPM of {}".format(top_feature.split("_")[0]))
    else:
        ax.set_xlabel("Mutation status of {}".format(top_feature))
    ax.set_ylabel("Gene effects (viability) of {}".format(knockout))
    plt.title("pcc: {}".format(r))
    plt.show()

def prompt_input(M,knockout,k,feature_type="gene",importances=False):
    if feature_type == "genes" and importances:
        print("ERROR: can only get importances for features not genes")
        return
    plot_feature_importances(knockout,M)

    data = get_top_indicators(M,k=k,feature_type=feature_type,importances=importances)
    if not importances:
        data = ", ".join(data)
    print("Top {} most important {}: \n".format(k,feature_type) + data)

    # plot_pcc(knockout,M)

# %%
knockout = "ARID1B"
feature_type = "features"                              ## feature or genes
importances = True                                 ## print feature importances?
k = 10 ## how many top features/genes
prompt_input(feature_importance_df_v3[knockout],knockout,k,feature_type,importances)