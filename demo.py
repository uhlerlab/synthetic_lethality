
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hickle as hkl
from scipy.stats import zscore

embedding = hkl.load("datasets/final_X_tcga_processed.hkl")
gene_effects_df = hkl.load("datasets/CRISPRGeneEffect_processed.hkl")
feature_importance_df = hkl.load("datasets/feature_importances.hkl")

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

def generate_inset_plots(M,knockout):
    M = M.sort_values(ascending=False)
    feature_importances = M.to_numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    freq, bins, patches = ax.hist(feature_importances)
    ax.set_title("{}".format(knockout))
    ax.set_xlabel("Feature Weights")
    ax.set_ylabel("Frequency (log scale)")
    plt.yscale('log')

    top_feature = M.index[0]

    bin_centers = np.diff(bins)*0.5 + bins[:-1]
    plt.annotate("{}".format(M.index[0]),
                xy = (bin_centers[-1], int(freq[-1]) + 0.1),             # top left corner of the histogram bar
                xytext = (0,0.2),             # offsetting label position above its bar
                textcoords = "offset points", # Offset (in points) from the *xy* value
                ha = 'center', va = 'bottom',
                fontsize=8,
                weight='bold'
                )

    left, bottom, width, height = [0.4, 0.35, 0.48, 0.48]
    ax2 = fig.add_axes([left, bottom, width, height])

    size = 8
    font = {'size': size}

    exp_feature = top_feature.split("_")[-1] == "exp"

    if exp_feature:
        ind = np.abs(zscore(embedding[top_feature].values)) < 3
        x = embedding[top_feature][ind]
        y = gene_effects_df[knockout][ind]
    else:
        x = embedding[top_feature]
        y = gene_effects_df[knockout]
    r = np.corrcoef([x,y])[0,1]

    exp_color = "seagreen" if r > 0 else "darkorange"
    x = embedding[top_feature]
    y = gene_effects_df[knockout]
    ax2.scatter(x,y,c=exp_color if exp_feature else "darkviolet")
    if exp_feature:
        ax2.set_xlabel("Gene expression TPM of {}".format(top_feature.split("_")[0]),**font)
    else:
        ax2.set_xlabel("Mutation status of {}".format(top_feature),**font)

    if exp_feature:
        if r > 0:
            line_color = "limegreen"
        else:
            line_color = "saddlebrown"
    else:
        line_color = "violet"
    
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),zorder=10,color=line_color)
    ax2.set_ylabel("Gene effects (viability) of {}".format(knockout),**font)
    ax2.tick_params(axis='x', labelsize=size)
    ax2.tick_params(axis='y', labelsize=size)

    plt.title("PCC: {}".format(str(round(r, 3))),**font)

    plt.show()

def prompt_input(M,knockout,k,feature_type="gene",importances=False):
    if feature_type == "genes" and importances:
        print("ERROR: can only get importances for features not genes")
        return
    generate_inset_plots(M,knockout)
    data = get_top_indicators(M,k=k,feature_type=feature_type,importances=importances)
    if not importances:
        data = ", ".join(data)
    print("Top {} most important {}: \n".format(k,feature_type) + data)


# %%
knockout = "ARID1B"
feature_type = "features"                              ## feature or genes
importances = True                                 ## print feature importances?
k = 10 ## how many top features/genes
prompt_input(feature_importance_df[knockout],knockout,k,feature_type,importances)