import os
import pandas as pd
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.utils import resample
from sklearn import random
import time
from tqdm import tqdm
import argparse
pd.options.mode.chained_assignment = None  # default='warn'
    

seed = 43
np.random.seed(seed)
random.seed(seed)


def test_spot_fun(
        adata: sc.AnnData,
        clust_vr: str,
        n=1000, 
        ):

    start_time = time.time()
    cell_types = adata.obs[clust_vr].unique()
    # count_matrix = adata.raw.X
    count_matrix = adata.X.toarray() if isinstance(adata.X, csr_matrix) or issparse(adata.X) else adata.X # cells x genes
    cell_names = adata.obs_names
    spots = []
    labels = []

    for i in tqdm(range(n), desc="Generating synthetic test spots..."):
        cell_pool = resample(cell_names, replace=False, n_samples=np.random.randint(2, 11))
        pos = [list(cell_names).index(cell) for cell in cell_pool]
        label = adata.obs.loc[cell_pool].copy()
        label['weight'] = 1
        label = label.groupby(clust_vr)['weight'].sum().reset_index()
        label['name'] = "spot_"+ str(i).zfill(6)
        syn_spot = count_matrix[pos, :].sum(axis=0)
        if syn_spot.sum() > 25000:
            syn_spot = syn_spot * (20000 / syn_spot.sum()) # downsample
        syn_spot = csr_matrix(syn_spot)
        spots.append(syn_spot)
        labels.append(label)

    # process the synthetic spots matrix
    syn_spot_counts = vstack(spots) # spots x genes
    syn_spot_counts = pd.DataFrame(syn_spot_counts.toarray()) # spots x genes
    syn_spot_counts.index = [f"mixt_{i}" for i in range(1, syn_spot_counts.shape[0] + 1)]
    syn_spot_counts.columns = adata.var_names

    # transform the format into spots x celltype weights
    syn_spots_metadata = pd.concat(labels)
    syn_spots_metadata = syn_spots_metadata.pivot(index='name', columns=clust_vr, values='weight').fillna(0)
    syn_spots_metadata.columns.name = None
    syn_spots_metadata.index = [f'mixt_{i}' for i in range(1, len(syn_spots_metadata) + 1)]
    syn_spots_metadata = syn_spots_metadata.div(syn_spots_metadata.sum(axis=1), axis=0)

    print(f"Generation of {n} test spots took {round(time.time() - start_time, 2)} seconds")

    return syn_spot_counts, syn_spots_metadata


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description='Generate synthetic spatial data')
    parser.add_argument('--num_spots', type=int, default=3000, help='Number of synthetic spots to generate')
    parser.add_argument('--data_root', type=str, default='./data/PDAC-A', help='Type of data to generate')
    args = parser.parse_args()
    num_spots = args.num_spots
    data_root = args.data_root
    
    sc_data_path = f"{data_root}/sc_data.h5ad" # cells x genes
    st_data_path = f"{data_root}/st_data.h5ad" # spots x genes
    metadata_path = f"{data_root}/meta_data.csv"
    sel_features_path = f"{data_root}/feature_list.txt"

    inforDir = f"{data_root}/Infor_Data"
    dataDir = f"{data_root}/Datadir"

    # read
    sc_count = sc.read_h5ad(sc_data_path) # cells x genes
    st_count = sc.read_h5ad(st_data_path) # spots x genes
    sc_count.var_names_make_unique()
    st_count.var_names_make_unique()
    metadata = pd.read_csv(metadata_path, index_col=0)
    metadata["celltype"].fillna("Unknown", inplace=True)

    # feature selection
    sel_features = pd.read_csv(sel_features_path, header=None)[0]
    sel_features = st_count.var_names.intersection(sel_features)
    sel_features = sc_count.var_names.intersection(sel_features)

    sc_count = sc_count[:, sel_features]
    st_count = st_count[:, sel_features]
    
    st_count = pd.DataFrame(st_count.X.toarray(), index = st_count.obs_names, columns = st_count.var_names) if isinstance(st_count.X, csr_matrix) or issparse(st_count.X) else \
        pd.DataFrame(st_count.X, index = st_count.obs_names, columns = st_count.var_names)
    sc_count.obs = metadata
    # spot generation
    syn_st_count, syn_st_meta = test_spot_fun(adata=sc_count, clust_vr="celltype", n=num_spots)
    st_counts = [syn_st_count, st_count]
    
    N1 = st_counts[0].shape[0]
    N2 = st_counts[1].shape[0]
    ori_st_label = pd.concat([syn_st_meta] * (round(N2 / N1) + 1)).iloc[:N2]
    st_labels = [syn_st_meta, ori_st_label]

    # save
    os.makedirs(inforDir, exist_ok=True)

    for subdir in ['ST_count', 'ST_label']:
        os.makedirs(os.path.join(inforDir, subdir), exist_ok=True)

    for i in range(2):
        st_counts[i].to_csv(os.path.join(inforDir, f'ST_count/ST_count_{i+1}.csv'))
        st_labels[i].to_csv(os.path.join(inforDir, f'ST_label/ST_label_{i+1}.csv'))

