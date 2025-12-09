import argparse
import logging
import os
import numpy as np
import pybedtools
import pyBigWig
import pandas as pd
from typing import Optional
from pythonase.region import extend_regions_from_mid_points

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO)


def build_sc_mat(fragments_files: list[str], regions_df: pd.DataFrame,
                 labels: Optional[tuple] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build reference matrix from single-cell experiments

    Parameters
    ----------
    fragments_files : list[str]

    regions_df : pd.DataFrame

    labels : Optional[tuple]


    Returns
    -------
    sc_mat : pd.DataFrame
        row: cell
        col: region
    psb_mat : pd.DataFrame
        row: cell
        col: region
    """
    if regions_df.shape[1] != 3:
        raise ValueError("Regions dataframe should have exactly three columns.")
    ref_regions_bed = pybedtools.BedTool.from_dataframe(regions_df)
    per_cell_covs = []
    per_frag_file_covs = []

    for i, fragments_file in enumerate(fragments_files):
        logger.info(f"Processing {fragments_file}...")
        fragments = pd.read_csv(fragments_file, sep="\t", comment="#", header=None)
        if labels is not None:
            lib_pref = labels[i]
        else:
            lib_pref = os.path.splitext(os.path.split(fragments_file)[1])[0]
        if fragments.shape[1] != 5:
            raise ValueError("Fragments file should have exactly five columns.")
        pffc_df = ref_regions_bed.coverage(
            pybedtools.BedTool.from_dataframe(fragments),
            nonamecheck=True).to_dataframe(
            disable_auto_names=True, header=None)
        pffc_df["region"] = pffc_df[0] + ":" + pffc_df[1].map(str) + "-" + pffc_df[2].map(str)
        pffc_df[lib_pref] = pffc_df[3]
        pffc_df.set_index("region", inplace=True)
        per_frag_file_covs.append(pffc_df[[lib_pref, ]])

        processed_cells = 0
        for cell_barcode, fragments_in_cell in fragments.groupby(3):
            cell_fragments_bed = pybedtools.BedTool.from_dataframe(fragments_in_cell)
            # cov_df will have 7 columns:
            # 0~2: coordinate of the region as defined in regions_df
            # 3: # records in cell_fragments_bed overlap with this region
            #
            cov_df = ref_regions_bed.coverage(
                cell_fragments_bed, nonamecheck=True).to_dataframe(
                disable_auto_names=True, header=None)
            cov_df["region"] = cov_df[0] + ":" + cov_df[1].map(str) + "-" + cov_df[2].map(str)
            col_name = f"{lib_pref}-{cell_barcode}"
            cov_df[col_name] = cov_df[3]
            cov_df.set_index("region", inplace=True)
            per_cell_covs.append(cov_df[[col_name, ]])
            processed_cells += 1
        logger.info(f"{fragments_file} processed ({processed_cells} cells).")
    return pd.concat(per_cell_covs, axis=1).T, pd.concat(per_frag_file_covs, axis=1).T


def build_bulk_mat(pl_files: list[str], mn_files: list[str], sample_labels: list[str],
                   regions_df: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """
    Build bulk matrix

    Parameters
    ----------
    pl_files : list[str]

    mn_files : list[str]

    sample_labels : list[str]

    regions_df : pd.DataFrame


    Returns
    -------
    tuple[pd.DataFrame, ...]
        bk_mat : pd.DataFrame
            row: bulk sample
            col: region
        pl_df : pd.DataFrame
        mn_df : pd.DataFrame
    """
    pl_values = {}
    mn_values = {}
    merged_values = {}
    _regions_df = regions_df.copy()
    for pl_file, mn_file, sample_label in zip(pl_files, mn_files, sample_labels):
        with pyBigWig.open(pl_file) as pl_bw, pyBigWig.open(mn_file) as mn_bw:
            pl_values[sample_label] = _regions_df.apply(
                lambda x: np.abs(np.nan_to_num(
                    pl_bw.values(x[0], x[1], x[2])
                ).sum()), axis=1)
            mn_values[sample_label] = _regions_df.apply(
                lambda x: np.abs(np.nan_to_num(
                    mn_bw.values(x[0], x[1], x[2])
                ).sum()), axis=1)
            merged_values[sample_label] = pl_values[sample_label] + mn_values[sample_label]
    dfs = []
    for source in (merged_values, pl_values, mn_values):
        df = pd.DataFrame(source)
        df["region"] = _regions_df[0] + ":" + _regions_df[1].map(str) + "-" + _regions_df[2].map(str)
        df.set_index("region", inplace=True)
        df.index.name = None
        dfs.append(df.T)

    return tuple(dfs)


def main_logic(arguments: argparse.Namespace):
    region_df = pd.read_csv(arguments.regions, sep=",", comment="#", header=None)
    if args.window_size > 0:
        region_df = extend_regions_from_mid_points(
            region_df, (args.window_size // 2, args.window_size // 2),
            chromosome_size=args.chrom_size
        )
    if arguments.pos_only:
        region_df = region_df.loc[region_df[3] == 1]

    if arguments.blacklists is not None:
        if len(arguments.blacklists):
            blacklist = pybedtools.BedTool.cat(*[pybedtools.BedTool(f) for f in arguments.blacklists])
        else:
            blacklist = pybedtools.BedTool(arguments.blacklists[0])
        region_bed = pybedtools.BedTool.from_dataframe(region_df[[0, 1, 2]]).intersect(blacklist, v=True)
        region_df = region_bed.to_dataframe(disable_auto_names=True, header=None)

    # build single-cell reference
    logger.info("Building single-cell reference")
    sc_mat, psb_mat = build_sc_mat(
        fragments_files=args.fragments, regions_df=region_df[[0, 1, 2]], labels=args.latent_label)
    dest = os.path.join(args.save_to, "sc_mat.csv")
    sc_mat.to_csv(dest, header=True, index=True)
    logger.info(f"Writing single-cell reference to {dest}")
    t_dest = os.path.join(args.save_to, "sc_mat.t.csv")
    sc_mat.T.to_csv(t_dest, header=True, index=True)
    logger.info(f"Writing single-cell reference to {t_dest}")
    dest = os.path.join(args.save_to, "psb_mat.csv")
    logger.info(f"Writing single-cell (pseudo-bulk) reference to {dest}")
    psb_mat.to_csv(dest, header=True, index=True)
    t_dest = os.path.join(args.save_to, "psb_mat.t.csv")
    logger.info(f"Writing transposed single-cell (pseudo-bulk) reference to {t_dest}")
    psb_mat.T.to_csv(t_dest, header=True, index=True)

    # build bulk mat
    logger.info("Building bulk mat")
    bulk_mat, pl_bulk_mat, mn_bulk_mat = build_bulk_mat(
        arguments.observable_pl, arguments.observable_mn,
        arguments.observable_label, region_df)
    dest = os.path.join(arguments.save_to, "bulk_mat.csv")
    logger.info(f"Writing bulk counts to {dest}")
    bulk_mat.to_csv(dest, header=True, index=True)
    t_dest = os.path.join(arguments.save_to, "bulk_mat.t.csv")
    logger.info(f"Writing bulk counts to {t_dest}")
    bulk_mat.T.to_csv(t_dest, header=True, index=True)
    pl_bulk_mat.to_csv(os.path.join(arguments.save_to, "bulk_mat.pl.csv"), header=True, index=True)
    pl_bulk_mat.T.to_csv(os.path.join(arguments.save_to, "bulk_mat.pl.t.csv"), header=True, index=True)
    mn_bulk_mat.to_csv(os.path.join(arguments.save_to, "bulk_mat.mn.csv"), header=True, index=True)
    mn_bulk_mat.T.to_csv(os.path.join(arguments.save_to, "bulk_mat.mn.t.csv"), header=True, index=True)

    # ground truth if exists
    if arguments.latent_pl is not None and arguments.latent_mn is not None:
        n_cell_types = len(arguments.fragments)
        for i, l in enumerate(arguments.observable_label):
            gt_mat, pl_gt_mat, mn_gt_mat = build_bulk_mat(
                arguments.latent_pl[i * n_cell_types: (i + 1) * n_cell_types],
                arguments.latent_mn[i * n_cell_types: (i + 1) * n_cell_types],
                arguments.latent_label[i * n_cell_types: (i + 1) * n_cell_types], region_df)
            dest = os.path.join(arguments.save_to, "ground_truth.csv" if i == 0 else f"ground_truth.{l}.csv")
            t_dest = os.path.join(arguments.save_to, "ground_truth.t.csv" if i == 0 else f"ground_truth.{l}.t.csv")
            logger.info(f"Writing ground truth to {dest} and {t_dest}")
            gt_mat.to_csv(dest, header=True, index=True)
            gt_mat.T.to_csv(t_dest, header=True, index=True)
            pl_gt_mat.T.to_csv(
                os.path.join(arguments.save_to, "ground_truth.pl.t.csv"), header=True, index=True)
            mn_gt_mat.T.to_csv(
                os.path.join(arguments.save_to, "ground_truth.mn.t.csv"), header=True, index=True)
            if len(arguments.latent_pl) < n_cell_types * len(arguments.observable_label):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("regions", help="Regions to be deconvoluted")
    parser.add_argument("--save-to", help="Save prepared files to this directory")
    parser.add_argument("--fragments", nargs="+", required=True, help="Fragment file(s) for each condition")
    parser.add_argument("--observable-pl", nargs="+", required=True,
                        help="Observed signals on the forward strand for each bulk sample")
    parser.add_argument("--observable-mn", nargs="+", required=True,
                        help="Observed signals on the reverse strand for each bulk sample")
    parser.add_argument("--observable-label", nargs="+", required=True,
                        help="Labels of observed signals for each bulk sample")
    parser.add_argument("--latent-pl", nargs="*", required=False,
                        help="Latent signals on the forward strand for each condition")
    parser.add_argument("--latent-mn", nargs="*", required=False,
                        help="Latent signals on the reverse strand for each condition")
    parser.add_argument("--latent-label", nargs="*", required=False,
                        help="Labels for latent observations")
    parser.add_argument("--all-regions", dest="pos_only", action="store_false",
                        help="Option to use all regions, by default only peak regions will be used")
    parser.add_argument("--blacklists", nargs="*", help="Regions to be excluded")
    parser.add_argument("--window-size", default=0, type=int,
                        help="If you want to center and re-extend regions, assign a greater than 0 to this argument")
    parser.add_argument("--chrom-size", type=str,
                        help="Path to a tab-separated file which stores sizes of chromosomes")

    args = parser.parse_args()

    # check whether files exist
    def _check_file(file_path):
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return True
        else:
            logger.exception(f"File {file_path} doesn't exist!")
            return False
    if not all((all(map(_check_file, args.fragments)),
                all(map(_check_file, args.latent_pl)),
                all(map(_check_file, args.latent_mn)),
                all(map(_check_file, args.observable_pl)),
                all(map(_check_file, args.observable_mn)),
                _check_file(args.chrom_size))):
        parser.error("Any file specified in --fragments, --latent-pl, --latent-mn, --observable-pl, "
                     "--observable-pl, or --chrom-size must be accessible.")

    input_numbers = (len(args.latent_label), len(args.latent_pl), len(args.latent_mn), len(args.fragments))
    if len(set(input_numbers)) > 1:
        parser.error(f"--fragments ({len(args.fragments)}), --latent-pl ({len(args.latent_pl)}), "
                     f"--latent-mn ({len(args.latent_mn)}), and --latent-label ({len(args.latent_label)}) "
                     "should have the same amount of values")
    if args.latent_pl is not None and args.latent_mn is not None:
        if not len(args.latent_pl) == len(args.latent_mn) == len(args.latent_label):
            parser.error(f"--latent-pl ({len(args.latent_pl)}), --latent-mn ({len(args.latent_mn)}), "
                         f"and --latent-label ({len(args.latent_label)}) should have the same amount of values.")
        if (len(args.latent_pl) % len(args.fragments) != 0 or len(args.latent_pl) // len(args.fragments) not in
                (1, len(args.observable_label))):
            parser.error("Number of --latent-pl / --latent-mn must be 1*--fragments or n_mixtures*--fragments")
        if len(args.latent_pl) // len(args.fragments) > 1:
            logger.warning("Latent files for more than one mixture are provided. "
                           "They will be evaluated on the same set of reference regions.")
    logger.info(args)
    pybedtools.set_tempdir(os.getcwd())

    try:
        main_logic(args)
    except Exception as e:
        logger.exception(e)
        raise
    finally:
        pybedtools.cleanup()
