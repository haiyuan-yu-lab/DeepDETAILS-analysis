import argparse
import logging
import os.path

import numpy as np
import pandas as pd
from typing import Optional
from Deconvolution.BLADE import Framework


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO)


def run_blade(mean: np.ndarray, sd: np.ndarray, y: np.ndarray, n_rep: int = 3, n_repfinal: int = 10,
              n_job: int = 1, ind_marker: Optional[list] = None):
    """
    Run BLADE

    Parameters
    ----------
    mean : np.ndarray
        a n_gene by n_cell matrix contains average gene expression profiles per cell type
        (a signature matrix) in log-scale
    sd : np.ndarray
        a n_gene by n_cell matrix contains standard deviation per gene per cell type
        (a signature matrix of gene expression variability).
    y : np.ndarray
        a n_gene by n_sample matrix contains bulk gene expression data.
        This should be in linear-scale data without log-transformation.
    n_rep : int
        Number of repeat for evaluating each parameter configuration.
    n_repfinal : int
        Number of repeated optimizations for the final parameter set.
    n_job : int
        Number of parallel jobs.
    ind_marker : Optional[list]
        Index for marker genes. By default, [True]*n_gene (all genes used without filtering).
        For the genes with False they are excluded in the first phase (Empirical Bayes) for
        finding the best hyperparameters.

    Returns
    -------
    fractions : np.ndarray
        fractions of different cell types, should be a n_sample by n_cell matrix contains
        estimated fraction of each cell type in each sample
    hires_gep : np.ndarray
        estimated gene expression levels of each gene in each cell type for each sample
        should be a n_sample by n_gene by n_cell multidimensional array
    group_gep : np.ndarray
        estimated gene expression profile per cell type, we can simply take an average across the samples.
        should be a n_gene by n_cell matrix
    """
    # n_genes should be consistent across mean, sd, and Y
    assert mean.shape[0] == sd.shape[0] == y.shape[0]
    # n_cell should be consistent across mean and sd
    assert mean.shape[1] == sd.shape[1]

    # run BLADE
    hyperpars = {
        'Alpha': [1, 10],
        'Alpha0': [0.1, 0.5, 1, 5, 10],
        'Kappa0': [1, 0.5, 0.1],
        'SY': np.sqrt([0.1, 0.5, 1, 1.5, 2])
    }

    final_obj, best_obj, best_set, outs = Framework(
        mean, sd, y, Ind_Marker=ind_marker,
        Alphas=hyperpars['Alpha'], Alpha0s=hyperpars['Alpha0'],
        Kappa0s=hyperpars['Kappa0'], SYs=hyperpars['SY'],
        Nrep=n_rep, Njob=n_job, Nrepfinal=n_repfinal)

    # Obtaining estimates

    # fractions of different cell types, should be a n_sample by n_cell matrix
    # contains estimated fraction of each cell type in each sample
    fractions = final_obj.ExpF(final_obj.Beta)

    # high-resolution mode purification
    # final_obj.Nu should be a n_sample by n_gene by n_cell multidimensional array
    # contains estimated gene expression levels of each gene in each cell type for each sample.
    hires_gep = final_obj.Nu

    # group mode purification
    # To obtain an estimated gene expression profile per cell type, we can simply take an average across the samples.
    group_gep = np.mean(final_obj.Nu, 0)

    return fractions, hires_gep, group_gep


def prep_data(sc_ref: str, mixtures: str, bulk_col_slice: Optional[tuple[int, int]]):
    """
    Prepare data for BLADE

    Parameters
    ----------
    sc_ref : str
        Path to the single cell reference matrix
    mixtures : str
        Path to the mixture matrix
    bulk_col_slice : Optional[tuple[int, int]]
        Start and end range for

    Returns
    -------
    sc_mu_ref_df : pd.DataFrame
        A N_gene by N_cell matrix contains average gene expression profiles per cell type (a signature matrix) in log-scale. (X)
    sc_std_ref_df : pd.DataFrame
        A N_gene by N_cell matrix contains standard deviation per gene per cell type (a signature matrix of gene expression variability). (stdX)
    mixtures_df : pd.DataFrame
        A N_gene by N_sample matrix contains bulk gene expression data. This should be in linear-scale data without log-transformation. (Y)
    """
    # sc_ref is stored in a csv file and the loaded dataframe should look like the following
    """
                        Caco2-AAACGAAAGGCAGATC-1  Caco2-AAACGAAAGTGATCTC-1  Caco2-AAACTGCAGGACTAGC-1  ...  MCF7-TTTGGTTAGTGTTCCA-1  MCF7-TTTGGTTTCCTAAGTG-1  MCF7-TTTGTGTGTCCCTAAA-1
region                                                                                            ...
chr1:10260-11261                           0                         0                         0  ...                        0                        0                        0
chr1:10295-11296                           0                         0                         0  ...                        0                        0                        0
chr1:778260-779261                         0                         0                         0  ...                        0                        1                        0
chr1:804350-805351                         0                         0                         0  ...                        0                        0                        0
chr1:826315-827316                         0                         0                         0  ...                        0                        1                        0
    """
    sc_ref_df = pd.read_csv(sc_ref, index_col=0)
    sc_ref_df.columns = sc_ref_df.columns.str.split("-").map(lambda x: x[0])

    per_type_cols = {k: np.where(sc_ref_df.columns == k)[0] for k in sc_ref_df.columns.unique()}

    agg_sc_ref_mu_cols = []
    agg_sc_ref_std_cols = []
    agg_sc_ref_col_names = []

    for ct, index in per_type_cols.items():
        cell_type_df = sc_ref_df.iloc[:, index]
        agg_sc_ref_mu_cols.append(cell_type_df.mean(axis=1))
        agg_sc_ref_std_cols.append(cell_type_df.std(axis=1))
        agg_sc_ref_col_names.append(ct)

    # sc_mu_ref_df (X)
    sc_mu_ref_df = np.log1p(pd.concat(agg_sc_ref_mu_cols, axis=1))
    sc_mu_ref_df.columns = agg_sc_ref_col_names

    # sc_std_ref_df (stdX)
    sc_std_ref_df = pd.concat(agg_sc_ref_std_cols, axis=1)
    sc_std_ref_df.columns = agg_sc_ref_col_names

    # build the mixture matrix (Y)
    # mixtures are stored in csv file and the loaded dataframe should look like the following
    """
                     5D1_1   5D1_2   5D1_3   5D1_4   5D1_5   5D1_6   5D1_7   5D1_8   5D1_9  5D1_10
chr1:604998-605999    77.0    83.0    83.0    75.0    79.0    84.0    95.0    79.0    75.0    84.0
chr1:778268-779269  2744.0  2752.0  2669.0  2730.0  2744.0  2652.0  2646.0  2703.0  2717.0  2720.0
chr1:779345-780346    30.0    24.0    22.0    26.0    23.0    29.0    28.0    25.0    32.0    23.0
chr1:826754-827755  1063.0  1053.0  1066.0  1053.0  1025.0  1001.0  1037.0  1030.0  1043.0  1000.0
chr1:826871-827872  1065.0  1056.0  1072.0  1052.0  1033.0  1002.0  1042.0  1035.0  1044.0  1001.0
    """
    mixtures_df = pd.read_csv(mixtures, index_col=0)
    # select a range of columns from all mixtures if --bulk-col-slice is specified
    if bulk_col_slice is not None and all([i <= mixtures_df.shape[1] for i in bulk_col_slice]):
        mixtures_df = mixtures_df[mixtures_df.columns[bulk_col_slice[0]:bulk_col_slice[1]]]

    return sc_mu_ref_df, sc_std_ref_df, mixtures_df


def main(arg):
    # get counts matrices
    x_df, std_df, y_df = prep_data(arg.sc_ref, arg.mixtures, arg.bulk_col_slice)

    # use dataframe.values because BLADE doesn't take DataFrames
    if arg.ind_marker and os.path.exists(arg.ind_marker):
        logger.info("Signature specified, loading...")
        sig_df = pd.read_csv(arg.ind_marker, sep="\t", index_col=0)
        if arg.signature_name_mapping and os.path.exists(arg.signature_name_mapping):
            row_mapping = pd.read_csv(arg.signature_name_mapping, index_col=0).set_index("mapped").to_dict()["original"]
            sig_df.index = sig_df.index.map(row_mapping)
        ind_marker = x_df.index.isin(sig_df.index).tolist()
    else:
        ind_marker = None
    logger.info(f"Shapes: mean df: {x_df.shape}\tstd df: {std_df.shape}\ty_df: {y_df.shape}")
    frac, hires, group = run_blade(x_df.values, std_df.values, y_df.values, ind_marker=ind_marker,
                                   n_rep=arg.n_rep, n_repfinal=arg.n_rep_final, n_job=arg.n_job)
    np.savez("out.npz", frac=frac, group=group, hires=hires)

    # export purified counts matrices
    for i, f in enumerate(frac):
        sample_name = y_df.columns[i]
        # raw predictions
        # - group mode
        group_df = pd.DataFrame(group, index=x_df.index, columns=x_df.columns)
        group_df.to_csv(f"Group.{sample_name}.csv.gz")

        # - high-resolution mode
        hires_df = pd.DataFrame(hires[i, :, :], index=x_df.index, columns=x_df.columns)
        hires_df.to_csv(f"HiRes.{sample_name}.csv.gz")
        # n_gene by n_cell

        # scale counts by their fraction so that the evaluation is consistent throughout the study
        # - group mode
        scaled_group_df = pd.DataFrame(group * f, index=x_df.index, columns=x_df.columns)
        scaled_group_df.to_csv(f"Group.scaled.{sample_name}.csv.gz")

        # - high-resolution mode
        scaled_hires_df = pd.DataFrame(f * hires[i, :, :], index=x_df.index, columns=x_df.columns)
        scaled_hires_df.to_csv(f"HiRes.scaled.{sample_name}.csv.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sc-ref", type=str, required=True,
                        help="Single cell reference sample file, where each row is a gene/region "
                             "and its expression values, and each column represents a single cell.")
    parser.add_argument("--mixtures", type=str, required=True,
                        help="Gene expression profile (GEP) matrix for the mixtures (bulk RNA-seq samples)."
                             "Rows for genes/regions, col 1 for gene name/region ID, col 2 for the mixture")
    parser.add_argument("--bulk-col-slice", nargs=2, type=int, required=False)

    grp = parser.add_argument_group("BLADE specific options")
    grp.add_argument("--n-rep", default=3, type=int,
                     help="Number of repeat for evaluating each parameter configuration.")
    grp.add_argument("--n-rep-final", default=10, type=int,
                     help="Number of repeated optimizations for the final parameter set.")
    grp.add_argument("--n-job", default=16, type=int,
                     help="Number of parallel jobs.")
    grp.add_argument("--signatures", dest="ind_marker", type=str,
                     help="A tsv file where the first column will be used as signatures")
    grp.add_argument("--signature-name-mapping", type=str,
                     help="Mapping signature row names to a new space. This is helpful when using signatures "
                          "identified by CIBERSORTx since we removed special characters.")

    args = parser.parse_args()

    logger.info(args)
    main(args)
