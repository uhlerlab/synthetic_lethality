import numpy as np
import pandas as pd
import hickle as hkl
from numpy.linalg import solve, svd, norm
import matplotlib.pyplot as plt
import torch

def make_cell_embedding():
    prefix = "datasets/2023/"
    damaging_file = prefix+"OmicsSomaticMutationsMatrixDamaging.csv"
    hotspot_file = prefix+"OmicsSomaticMutationsMatrixHotspot.csv"
    expression_file = prefix+"OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    mutations_files = [damaging_file,hotspot_file]

    gene_effects_df = pd.read_csv("datasets/2023/CRISPRGeneEffect.csv",index_col=0)
    gene_effects_df.columns = [x.split()[0] for x in gene_effects_df.columns]
    gene_effects_df = gene_effects_df.set_index('DepMap_ID')

    embedding = None
    for file in mutations_files:
        mutations_df = pd.read_csv(file)
        mutations_df = mutations_df.set_index(mutations_df.columns[0]).fillna(0).astype(int)
        mutations_df.columns = [x.split()[0] for x in mutations_df.columns]
        if embedding is None:
            embedding = mutations_df
        else:
            # bitwise or between the two types of mutations
            for gene in embedding.columns:
                for cell in embedding.index:
                    if gene in mutations_df.columns and cell in mutations_df.index:
                        embedding.at[cell,gene] = ((embedding.at[cell,gene]) | (mutations_df.at[cell,gene])).astype(int)
            embedding = embedding.merge(mutations_df[[x for x in mutations_df.columns if x not in embedding.columns]], left_index=True, right_index=True,
                                                                            how='outer').fillna(0)

    exp_df = pd.read_csv(expression_file)
    exp_df = exp_df.set_index(exp_df.columns[0])

    exp_df.columns = [x.split()[0] + "_exp" for x in exp_df.columns]
    common_cells = gene_effects_df.index.intersection(embedding.index).intersection(exp_df.index)
    embedding = embedding.loc[common_cells]
    gene_effects_df = gene_effects_df.loc[common_cells]
    exp_df = exp_df.loc[common_cells]

    # merge two components
    embedding = embedding.merge(exp_df, left_index=True, right_index=True, how='inner')
    embedding = embedding.dropna(axis=1,how='all').fillna(0)

    # filter for TCGA columns (data from https://xenabrowser.net/datapages/)
    directory = "tcga/"
    disease = "BRCA" #random
    tcga_mut_cols = pd.read_csv(directory+disease+"/"+"mc3_gene_level_{}_mc3_gene_level.txt".format(disease),sep="\t", index_col=0).T.columns.tolist()
    tcga_exp_cols = pd.read_csv(directory+disease+"/"+"TCGA.{}.sampleMap_HiSeqV2".format(disease),sep="\t", index_col=0).T.columns.tolist()
    tcga_exp_cols = [col + "_exp" for col in tcga_exp_cols]
    tcga_cols = tcga_mut_cols + tcga_exp_cols
    common_cols = list(set(tcga_cols) & set(embedding.columns))
    embedding = embedding[common_cols]

    exp_cols = [x for x in common_cols if x.split("_")[-1] == "exp"]

    embedding = embedding.fillna(0)

    # z-score exp columns
    std_val = embedding[exp_cols].std(axis=0).replace(0,1)
    embedding[exp_cols]= (embedding[exp_cols] - embedding[exp_cols].mean(axis=0))/std_val
    # normalize embedding (if training model)
    embedding /= norm(embedding, axis=1).reshape(-1, 1)

    return embedding, gene_effects_df