import scanpy as sc
import pandas as pd

def feature_selection():
    # Read the gene expression data and metadata
    sc_adata = sc.read_h5ad(sc_h5_data_path)
    sc_adata_meta = pd.read_csv(sc_csv_meta_path, index_col=0)
    sc_adata_meta_ = sc_adata_meta.loc[sc_adata.obs.index,]
    sc_adata.obs = sc_adata_meta_
    sc_adata.var_names_make_unique()  
    print("Reading finished")

    # Preprocessing
    sc_adata.var['mt'] = sc_adata.var_names.str.startswith('Mt-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(sc_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    # mito_genes = sc_adata.var_names.str.startswith('mt-')
    # sc_adata = sc_adata[:, ~mito_genes]
    sc.pp.normalize_total(sc_adata)
    # sc.pp.log1p(sc_adata)
    # sc.pp.highly_variable_genes(sc_adata, flavor='seurat_v3', n_top_genes=2000)
    sc.tl.pca(sc_adata, svd_solver='arpack')
    # print("Preprocessing finished")
    
    sc.tl.rank_genes_groups(sc_adata, 'celltype', method='wilcoxon')
    sc.pl.rank_genes_groups(sc_adata, n_genes=30, sharey=False)
    print("rank_genes_groups finished")

    genelists=sc_adata.uns['rank_genes_groups']['names']
    df_genelists = pd.DataFrame.from_records(genelists)

    num_markers=30
    res_genes = []
    for column in df_genelists.head(num_markers): 
        res_genes.extend(df_genelists.head(num_markers)[column].tolist())
    res_genes_ = list(set(res_genes))
    from tqdm import tqdm
    with open(txt_path, 'w') as f:
        # Write each gene in the list as a string on a new line
        for item in tqdm(res_genes_, desc='Writing'):
            f.write(str(item) + '\n')

if __name__ == '__main__':
    folder = 'PDAC-A'
    txt_path = f'./data/{folder}/feature_list.txt'
    sc_h5_data_path = f'./data/{folder}/sc_data.h5ad'
    sc_csv_meta_path = f'./data/{folder}/meta_data.csv'
    feature_selection()
