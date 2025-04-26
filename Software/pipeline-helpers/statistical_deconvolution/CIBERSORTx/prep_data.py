import argparse
import csv
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sc-ref", type=str, required=True,
                        help="Single cell reference sample file, where each row is a gene "
                             "and its expression values, and each column represents a single cell. Each cell must be "
                             "assigned a cell phenotype, and there should be a minimum number of cells per phenotype (3)"
                             "(this minimmum number is set by the 'replicates' option under 'Single Cell Input Options', default is 5).")
    parser.add_argument("--mixtures", type=str, nargs="+", required=True,
                        help="Gene expression profile (GEP) matrix for the mixtures (bulk RNA-seq samples)."
                             "Rows for genes/regions, col 1 for gene name/region ID, col 2 for the mixture")
    parser.add_argument("--mixture-labels", type=str, nargs="+", required=False,
                        help="Labels for each mixture. This is only required when mixtures are stored in separate files.")
    parser.add_argument(
        "-t", "--transpose-ref", action="store_true",
        help="Transpose the reference dataframe before using."
    )
    parser.add_argument("--aggregate-refs-to", default=0, type=int)
    parser.add_argument("--bulk-col-slice", nargs=2, type=int, required=False)

    args = parser.parse_args()
    logger.info(args)

    # build the mixture matrix
    if len(args.mixtures) > 1:  # mixtures in different files
        if len(args.mixtures) != len(args.mixture_labels):
            logger.error("Each mixture should have its corresponding label")
        _mixtures = []
        _mixture_labels = []
        for l, m_file in zip(args.mixture_labels, args.mixtures):
            _df = pd.read_csv(m_file, index_col=0)
            _mixtures.append(_df)
            _mixture_labels.append(l)
        mixtures_df = pd.concat(_mixtures, axis=1)
        mixtures_df.columns = _mixture_labels
    else:  # mixtures in a single file
        mixtures_df = pd.read_csv(args.mixtures[0], index_col=0)
    # select a range of columns from all mixtures if --bulk-col-slice is specified
    if args.bulk_col_slice is not None and all([i <= mixtures_df.shape[1] for i in args.bulk_col_slice]):
        mixtures_df = mixtures_df[mixtures_df.columns[args.bulk_col_slice[0]:args.bulk_col_slice[1]]]

    # build the reference matrix
    _msg_min_ct_warning = "According to the authors, there should be at least three cells per phenotype!"

    sc_ref_df = pd.read_csv(args.sc_ref, index_col=0)
    if args.transpose_ref:
        sc_ref_df = sc_ref_df.T
    # simplify column names to only cell type/phenotype
    sc_ref_df.columns = sc_ref_df.columns.str.split("-").map(lambda x: x[0])
    per_type_counts = sc_ref_df.columns.value_counts()
    min_counts = per_type_counts.min()
    if min_counts < 3:
        logger.warning(_msg_min_ct_warning)

    # scATAC produces binary-like counts matrix.
    # Do we need to aggregate some cells sharing the same type labels
    # so that the counts matrix look like that from scRNA-seq more
    if args.aggregate_refs_to > 0:
        if args.aggregate_refs_to < 3:
            logger.warning(_msg_min_ct_warning)
        sample_size = int(min_counts // args.aggregate_refs_to)
        per_type_cols = {k: np.where(sc_ref_df.columns == k)[0] for k in sc_ref_df.columns.unique()}

        agg_sc_ref_cols = []
        agg_sc_ref_col_names = []

        for ct, index in per_type_cols.items():
            for probe in np.random.choice(index, size=sample_size * args.aggregate_refs_to, replace=False).reshape(
                    -1, sample_size):
                agg_sc_ref_cols.append(sc_ref_df.iloc[:, probe].sum(axis=1))
                agg_sc_ref_col_names.append(ct)
        sc_ref_df = pd.concat(agg_sc_ref_cols, axis=1)
        sc_ref_df.columns = agg_sc_ref_col_names

    # sanity check: make sure mixtures/bulks have the same genes/regions as single-cell references
    if sorted(sc_ref_df.index) != sorted(mixtures_df.index):
        logger.error("sc_ref_df and mixtures_df have different indexes")

    # rename row names to make sure CIBERSORTx will not get panicked by special characters
    prev_rows = sc_ref_df.index.values.copy()
    new_rows = [f"R{i}" for i in range(sc_ref_df.shape[0])]
    sc_ref_df.index = new_rows
    row_mapping_df = pd.DataFrame({"original": prev_rows, "mapped": new_rows})
    row_mapping = row_mapping_df.set_index("original").to_dict()["mapped"]
    mixtures_df.index = mixtures_df.index.map(row_mapping)

    mixtures_df.index.name = "region"
    sc_ref_df.index.name = "region"

    # export the mixture mat
    mixtures_df.to_csv("bulk.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    # export the single-cell reference mat
    sc_ref_df.to_csv("refsample.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    # export row name mapping
    row_mapping_df.to_csv("row_mapping.csv")
