import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, Callable


def counts_aggregation(in_df: pd.DataFrame, aggregation_method: str | Callable[..., Any]) -> pd.DataFrame:
    """Aggregate sample columns into one value per feature and condition.

    Column names must look like ``condition_replicate`` (single underscore
    separates condition from replicate). Values are melted, grouped by
    feature index and condition, aggregated, then pivoted back to wide form
    with one column per condition.

    Parameters
    ----------
    in_df : pd.DataFrame
        Counts with feature IDs as the index and sample names as columns.
    aggregation_method : str or callable
        Passed to :meth:`pandas.core.groupby.DataFrameGroupBy.agg` (e.g.
        ``'sum'`` or ``'mean'``).

    Returns
    -------
    pd.DataFrame
        Same row index as input features; columns are condition names.
    """
    melted_df = in_df.reset_index().melt(id_vars="index", var_name="sample")
    cond_df = melted_df["sample"].str.split("_", expand=True)
    melted_df["condition"] = cond_df[0]
    melted_df["rep"] = cond_df[1]

    pivoted_df = melted_df[
        ["index", "condition", "value"]
    ].groupby(
        ["index", "condition"]
    ).agg(
        aggregation_method
    ).reset_index().set_index(
        "index"
    ).pivot(
        columns="condition"
    )
    df_flat = pivoted_df.droplevel(0, axis=1).reset_index()
    df_flat.columns.name = None
    df_flat.set_index("index", inplace=True)
    return df_flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--real-counts", required=True, type=str)
    parser.add_argument("-I", "--dec-counts", required=True, type=str)
    parser.add_argument("-d", "--deg", required=True, type=str)
    parser.add_argument("-c", "--contrast", required=True, type=str, nargs=2)
    parser.add_argument("--delta", default=0.01, type=float)
    parser.add_argument("--aggregation-method-real", default="sum")
    parser.add_argument("--aggregation-method-dec", default="mean")

    args = parser.parse_args()

    real_counts_df = pd.read_csv(args.real_counts, index_col=0)
    deconvolved_counts_df = pd.read_csv(args.dec_counts, index_col=0)

    deg_df = pd.read_csv(args.deg, index_col=0)
    print(deg_df.shape)

    agg_real_counts_df = counts_aggregation(real_counts_df, args.aggregation_method_real)
    agg_real_counts_df += args.delta
    agg_dec_counts_df = counts_aggregation(deconvolved_counts_df, args.aggregation_method_dec)
    agg_dec_counts_df += args.delta

    for c in args.contrast:
        assert c in agg_real_counts_df.columns
        assert c in agg_dec_counts_df.columns

    filtered_agg_real_counts_df = agg_real_counts_df.loc[deg_df.index].copy()
    filtered_agg_dec_counts_df = agg_dec_counts_df.loc[deg_df.index].copy()
    filtered_agg_real_counts_df["LFC"] = np.log2(filtered_agg_real_counts_df[args.contrast[0]] / filtered_agg_real_counts_df[args.contrast[1]])
    filtered_agg_dec_counts_df["LFC"] = np.log2(filtered_agg_dec_counts_df[args.contrast[0]] / filtered_agg_dec_counts_df[args.contrast[1]])

    result_df = pd.DataFrame({
        "Observation": filtered_agg_real_counts_df["LFC"],
        "Deconvolved": filtered_agg_dec_counts_df["LFC"],
    })

    result_df.to_csv("res.csv", index=True)
    print("Accuracy", (np.sign(result_df['Observation']) == np.sign(result_df['Deconvolved'])).sum() / result_df.shape[0])
    print("R", np.corrcoef(result_df["Observation"], result_df["Deconvolved"]))

    fig, ax = plt.subplots()
    ax = sns.scatterplot(x=filtered_agg_real_counts_df["LFC"], y=filtered_agg_dec_counts_df["LFC"], s=1, alpha=0.05, ax=ax)
    ax.axhline(0, lw=1, zorder=-1)
    ax.axvline(0, lw=1, zorder=-1)
    sns.despine()
    plt.savefig("FC.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
