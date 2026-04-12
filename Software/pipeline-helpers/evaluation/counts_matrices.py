import argparse
import logging
import os
import pybedtools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pythonase.region import extend_regions_from_mid_points
from torch import tensor
from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.functional.regression.concordance import concordance_corrcoef

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO)
pybedtools.set_tempdir(".")
_imf_dir = "intermediate_files"


def convert_int_cols_to_float(in_df: pd.DataFrame):
    """
    Convert all integer columns in a pandas DataFrame to float.

    This function iterates over each column in the input DataFrame. If the data type of the column is 'int64',
    it converts that column to 'float64' using the astype function.

    Parameters
    ----------
    in_df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    None. The function operates in-place on the input DataFrame.
    """
    for col in in_df.columns:
        if in_df[col].dtype == 'int64':
            in_df[col] = in_df[col].astype(float)


def prep_dfs(
    ref: str,
    preds: list[str],
    chrom_size: str = "hg38",
    transpose_ref: bool = False,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Prepare dataframes. This function realigns predicted regions based on their midpoints
    and only shared regions will be kept.

    Parameters
    ----------
    ref : str
        Path to the reference counts matrix
    preds : list[str]
        Paths to the predicted counts matrices
    chrom_size : str
        Path to the chromosome size file or name of the reference genome
    transpose_ref : bool
        Set it as True if you want to transpose the reference dataframe first
    random_seed : int
        Seed for :func:`numpy.random.Generator.lognormal` when imputing NaN, inf, or negative values

    Returns
    -------
    ref_df : pd.DataFrame
        Aligned reference DataFrame
    pred_dfs : list[pd.DataFrame]
        Aligned prediction DataFrame
    """
    _ref = pd.read_csv(ref, index_col=0)
    if transpose_ref:
        _ref = _ref.T
    logger.info(f"Shape of the transposed reference dataframe: {_ref.shape}")
    convert_int_cols_to_float(_ref)
    _ref.index = _ref.index.str.replace(":", ".").str.replace("-", ".")
    _ref_regions = extend_regions_from_mid_points(
        pd.DataFrame(pd.Series(_ref.index).str.split(".", expand=True).values), (1, 1), chrom_size)
    _ref.index = _ref_regions[0] + "." + _ref_regions[1].map(str) + "." + _ref_regions[2].map(str)

    rng = np.random.default_rng(random_seed)
    _preds = []
    for p_file in preds:
        _tmp_df = pd.read_csv(p_file, index_col=0)
        convert_int_cols_to_float(_tmp_df)
        logger.info(f"Shape of the prediction dataframe ({p_file}): {_tmp_df.shape}")

        _tmp_df.index = _tmp_df.index.str.replace(":", ".").str.replace("-", ".")
        _pred_regions = extend_regions_from_mid_points(
            pd.DataFrame(pd.Series(_tmp_df.index).str.split(".", expand=True).values), (1, 1), chrom_size)
        regions_as_index = _pred_regions[0] + "." + _pred_regions[1].map(str) + "." + _pred_regions[2].map(str)
        _tmp_df.index = regions_as_index
        if _tmp_df.shape[1] == _ref.shape[1] * 2:  # merge stranded counts
            _sl_tmp = pd.DataFrame(
                {f"{_ref.columns[i]}": _tmp_df[f"{i * 2}"] + _tmp_df[f"{i * 2 + 1}"] for i in range(_ref.shape[1])},
                index=regions_as_index, )
            _tmp_df = _sl_tmp

        # replace NaN values
        nan_stats = _tmp_df.isna().sum()
        if nan_stats.sum() > 0:
            # replace NaN values with random small values draw from a lognormal distribution LN(0, 1)
            _tmp_df = _tmp_df.map(lambda l: l if not np.isnan(l) else 0.1 * rng.lognormal(0, 1))
            logger.warning(f"NaNs in {p_file} (column-wise):\n{nan_stats}")
        # replace inf values
        inf_stats = np.isinf(_tmp_df).sum()
        if inf_stats.sum() > 0:
            # replace inf values with random small values draw from a lognormal distribution LN(0, 1)
            _tmp_df = _tmp_df.map(lambda l: l if not np.isinf(l) else 0.1 * rng.lognormal(0, 1))
            logger.warning(f"Infs in {p_file} (column-wise):\n{inf_stats}")
        # replace negative counts
        negative_stats = (_tmp_df < 0).sum()
        if negative_stats.sum() > 0:
            # replace negative values with random small values draw from a lognormal distribution LN(0, 1)
            _tmp_df = _tmp_df.map(lambda l: l if l >= 0 else 0.1 * rng.lognormal(0, 1))
            logger.warning(f"Negative values in {p_file} (column-wise):\n{negative_stats}")

        _preds.append(_tmp_df)
    # figure out the shared regions
    indices = [df.index for df in _preds]
    shared_regions = tuple(
        set(_ref.index.values).intersection(set(indices[0]).intersection(*indices)))
    logger.info(f"Shared regions left: {len(shared_regions)}")
    _aligned_preds = [p.loc[p.index.isin(shared_regions)] for p in _preds]
    return _ref.loc[_ref.index.isin(shared_regions)], _aligned_preds


def _preprocessing_and_sanity_check(df_lst: list[pd.DataFrame], ref: pd.DataFrame, log_transform: bool = True) -> tuple[
    list[pd.DataFrame], pd.DataFrame]:
    """Preprocessing and sanity check

    Reorder rows, check if columns are consistent, and apply log-transformation if requested

    Parameters
    ----------
    df_lst : list[pd.DataFrame]
        list of prediction DataFrames
    ref : pd.DataFrame
        Reference counts matrix
    log_transform : bool
        Set it to True if you want to apply log/log1p transformation

    Returns
    -------
    processed_df_list : list[pd.DataFrame]

    processed_ref_df : pd.DataFrame

    """
    if log_transform:
        _preds = [np.log1p(_df.loc[ref.index]) for _df in df_lst]
        _ref = np.log1p(ref)
    else:
        _preds = [_df.loc[ref.index] for _df in df_lst]
        _ref = ref
    assert all(all(c in _ref.columns for c in exp.columns) for exp in _preds)
    return _preds, _ref


def _rmse(a: np.ndarray, b: np.ndarray):
    """
    root mean-square error

    Parameters
    ----------
    a
    b

    Returns
    -------

    """
    return np.sqrt(np.mean((a - b) ** 2))


def get_shape_corrs(
    pred_df_lst: list[pd.DataFrame],
    ref: pd.DataFrame,
    exp_labels: list[str],
    log_transform: bool = True,
    ccc: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Calculate correlations for shape and relative scales

    Parameters
    ----------
    pred_df_lst : list[pd.DataFrame]
        List of predicted counts matrices
    ref : pd.DataFrame
        Reference counts matrix
    exp_labels : list[str]
        Labels for the predictions
    log_transform : bool
        Set it as True to apply log1p transformation to the counts matrices
    ccc : bool
        Use Lin's Concordance correlation coefficient instead of Pearson's R

    Returns
    -------
    results : pd.DataFrame
        cols: "Model", "Cluster", "R"
    intermediate_dfs : dict[str, pd.DataFrame]
        key : source of the predictions as specified in `exp_labels`
        values : DataFrame for the pred vs ref counts
    """
    if ccc:
        cor_func = concordance_corrcoef
    else:
        cor_func = pearson_corrcoef
    _preds, _ref = _preprocessing_and_sanity_check(pred_df_lst, ref, log_transform=log_transform)
    results = []
    intermediate_dfs = {}
    for i, label in enumerate(exp_labels):
        _cols = []
        _col_labels = []
        for c in _ref.columns:
            results.append((label, c, cor_func(tensor(_preds[i][c]), tensor(_ref[c])).item()))
            _cols.append(_ref[c])
            _col_labels.append(f"{c}_ref")
            _cols.append(_preds[i][c])
            _col_labels.append(f"{c}_pred")
        int_df = pd.concat(_cols, axis=1)
        int_df.columns = _col_labels
        intermediate_dfs[label] = int_df
    return pd.DataFrame(results, columns=["Model", "Cluster", "R"]), intermediate_dfs


def get_pseudorank_corrs(
    pred_df_lst: list[pd.DataFrame],
    ref: pd.DataFrame,
    exp_labels: list[str],
    log_transform: bool = True,
    ccc: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Calculate correlations for the "rank"s

    Parameters
    ----------
    pred_df_lst : list[pd.DataFrame]
        List of predicted counts matrices
    ref : pd.DataFrame
        Reference counts matrix
    exp_labels : list[str]
        Labels for the predictions
    log_transform : bool
        Set it as True to apply log1p transformation to the counts matrices
    ccc : bool
        Use Lin's Concordance correlation coefficient instead of Pearson's R

    Returns
    -------
    results : pd.DataFrame
        cols: "Region", "Model", "R"
    intermediate_dfs : dict[str, pd.DataFrame]
        key : source of the predictions as specified in `exp_labels`
        values : DataFrame for the pred vs ref counts
    """
    if ccc:
        cor_func = concordance_corrcoef
    else:
        cor_func = pearson_corrcoef
    _preds, _ref = _preprocessing_and_sanity_check(pred_df_lst, ref, log_transform=log_transform)

    results = []
    intermediate_dfs = {}
    for i, label in enumerate(exp_labels):
        for r in _ref.index:
            # region, exp_label, corr
            results.append((r, label, cor_func(tensor(_preds[i].loc[r]), tensor(_ref.loc[r])).item()))
        _pred_df = _preds[i].copy()
        _pred_df.columns = [f"{c}_pred" for c in _pred_df.columns]
        _ref_df = _ref.copy()
        _ref_df.columns = [f"{c}_ref" for c in _ref_df.columns]
        int_df = pd.concat([_pred_df, _ref_df], axis=1)
        int_df = int_df.reindex(sorted(int_df.columns), axis=1)
        intermediate_dfs[label] = int_df
    return pd.DataFrame(results, columns=["Region", "Model", "R"]), intermediate_dfs


def get_rmse(
    pred_df_lst: list[pd.DataFrame],
    ref: pd.DataFrame,
    exp_labels: list[str],
    log_transform: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Calculate RMSE

    Parameters
    ----------
    pred_df_lst : list[pd.DataFrame]
        List of predicted counts matrices
    ref : pd.DataFrame
        Reference counts matrix
    exp_labels : list[str]
        Labels for the predictions
    log_transform : bool
        Set it as True to apply log1p transformation to the counts matrices

    Returns
    -------
    results : pd.DataFrame
        cols: "Region", "Model", "RMSE"
    intermediate_dfs : dict[str, pd.DataFrame]
        key : source of the predictions as specified in `exp_labels`
        values : DataFrame for the pred vs ref counts
    """
    _preds, _ref = _preprocessing_and_sanity_check(pred_df_lst, ref, log_transform=log_transform)

    results = []
    intermediate_dfs = {}
    for i, label in enumerate(exp_labels):
        for r in _ref.index:
            # region, exp_label, corr
            results.append((r, label, _rmse(_preds[i].loc[r], _ref.loc[r])))
        _pred_df = _preds[i].copy()
        _pred_df.columns = [f"{c}_pred" for c in _pred_df.columns]
        _ref_df = _ref.copy()
        _ref_df.columns = [f"{c}_ref" for c in _ref_df.columns]
        int_df = pd.concat([_pred_df, _ref_df], axis=1)
        int_df = int_df.reindex(sorted(int_df.columns), axis=1)
        intermediate_dfs[label] = int_df
    return pd.DataFrame(results, columns=["Region", "Model", "RMSE"]), intermediate_dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions in the forms of counts matrices")
    parser.add_argument(
        "-r", "--ref", required=True,
        help="Path to the reference counts table. Rows for the clusters, cols for the regions.")
    parser.add_argument(
        "-p", "--preds", nargs="+", required=True,
        help="Paths to the predictions. Rows for the regions, cols for the clusters."
    )
    parser.add_argument(
        "-l", "--labels", nargs="+", required=True,
        help="Labels for the sources of each prediction"
    )
    parser.add_argument(
        "-o", "--output-prefix", required=True,
        help="Output prefix. Final outputs will be outputPrefix.type.suffix.ext"
    )
    parser.add_argument(
        "-c", "--chrom-size", required=False, default="hg38",
        help="Path to a tab-separated file which stores chromosome size info."
    )
    parser.add_argument(
        "-s", "--fig-size", required=False, default=(4, 2), nargs=2, type=float,
        help="Fig size"
    )
    parser.add_argument(
        "-t", "--transpose-ref", action="store_true",
        help="Transpose the reference dataframe before using. For backward compatibility."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for lognormal imputation of NaN, inf, and negative prediction values.",
    )
    args = parser.parse_args()

    if len(args.preds) != len(args.labels):
        raise parser.error(f"The number of predictions ({len(args.preds)}) and labels ({len(args.labels)}) must match")

    os.makedirs(_imf_dir, exist_ok=True)

    labels = [f"{l.split('_')[0]} ({l.split('_')[1]})" if l.count("_") == 1 else l for l in args.labels]
    ref_df, pred_dfs = prep_dfs(
        args.ref, list(args.preds), args.chrom_size, args.transpose_ref, random_seed=args.seed,
    )

    for suffix, operation in zip(("log", "raw"), (True, False)):
        shape_cor_df, sc_imdfs = get_shape_corrs(pred_dfs, ref_df,
                                                 [rf"{l}" if l.find("$") != -1 else l for l in labels],
                                                 log_transform=operation)
        shape_cor_df.to_csv(f"{args.output_prefix}.shape_corr.{suffix}.csv.gz")
        for k, v in sc_imdfs.items():
            v.to_csv(f"{_imf_dir}/{k}.{suffix}.sc.csv.gz")
        shape_ccc_df, sc_ccc_imdfs = get_shape_corrs(
            pred_dfs, ref_df, [rf"{l}" if l.find("$") != -1 else l for l in labels],
            log_transform=operation, ccc=True)
        shape_ccc_df.to_csv(f"{args.output_prefix}.shape_ccc.{suffix}.csv.gz")
        for k, v in sc_ccc_imdfs.items():
            v.to_csv(f"{_imf_dir}/{k}.{suffix}.sc_ccc.csv.gz")
        rank_cor_df, rc_imdfs = get_pseudorank_corrs(pred_dfs, ref_df,
                                                     [rf"{l}" if l.find("$") != -1 else l for l in labels],
                                                     log_transform=operation)
        for k, v in rc_imdfs.items():
            v.to_csv(f"{_imf_dir}/{k}.{suffix}.rc.csv.gz")
        rank_cor_df.to_csv(f"{args.output_prefix}.rank_corr.{suffix}.csv.gz")
        rank_ccc_df, rc_ccc_imdfs = get_pseudorank_corrs(
            pred_dfs, ref_df, [rf"{l}" if l.find("$") != -1 else l for l in labels],
            log_transform=operation, ccc=True)
        for k, v in rc_ccc_imdfs.items():
            v.to_csv(f"{_imf_dir}/{k}.{suffix}.rc.ccc.csv.gz")
        rank_ccc_df.to_csv(f"{args.output_prefix}.rank_ccc.{suffix}.csv.gz")

        rmse_df, rmse_imdfs = get_rmse(pred_dfs, ref_df,
                                       [rf"{l}" if l.find("$") != -1 else l for l in labels],
                                       log_transform=operation)
        for k, v in rmse_imdfs.items():
            v.to_csv(f"{_imf_dir}/{k}.{suffix}.rmse.csv.gz")
        rmse_df.to_csv(f"{args.output_prefix}.rmse.{suffix}.csv.gz")

        fig, axes = plt.subplots(1, 5, figsize=args.fig_size, sharey=False)
        sns.violinplot(data=shape_cor_df, x="Model", y="R", ax=axes[0], cut=0)
        axes[0].set_xlabel("")
        axes[0].set_ylim(-0.03, 1.01)
        axes[0].set_ylabel(r"$r$ (across TREs)", fontsize=7)

        sns.violinplot(data=rank_cor_df, x="Model", y="R", ax=axes[1], cut=0)
        axes[1].axhline(0, lw=axes[1].spines["top"].get_linewidth(), color="black", zorder=-1)
        axes[1].set_ylabel(r"$r$ (across cell types)", fontsize=7)
        axes[1].set_ylim(-1.01, 1.01)
        axes[1].set_xlabel("")

        sns.violinplot(data=shape_ccc_df, x="Model", y="R", ax=axes[2], cut=0)
        axes[2].set_xlabel("")
        axes[2].set_ylim(-0.03, 1.01)
        axes[2].set_ylabel(r"$CCC$ (across TREs)", fontsize=7)

        sns.violinplot(data=rank_ccc_df, x="Model", y="R", ax=axes[3], cut=0)
        axes[3].axhline(0, lw=axes[3].spines["top"].get_linewidth(), color="black", zorder=-1)
        axes[3].set_ylabel(r"$CCC$ (across cell types)", fontsize=7)
        axes[3].set_ylim(-1.01, 1.01)
        axes[3].set_xlabel("")

        sns.violinplot(data=rmse_df, x="Model", y="RMSE", ax=axes[4], cut=0)
        axes[4].set_ylabel("RMSE", fontsize=7)
        axes[4].set_xlabel("")
        sns.despine()
        for ax in axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.tick_params(axis="both", which="major", labelsize=6)
        axes[1].spines["bottom"].set_visible(False)
        axes[3].spines["bottom"].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{args.output_prefix}.{suffix}.pdf", transparent=True, bbox_inches="tight")
        plt.close()
