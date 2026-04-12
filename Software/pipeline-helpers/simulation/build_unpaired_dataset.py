import argparse
import logging
import os
import h5py
import pandas as pd
try:
    from deepdetails.protocols import prepare_dataset
except ImportError:
    from details.protocols import prepare_dataset
from pythonase.io import to_csv_with_comments

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO)


def create_link(source, target):
    if os.path.exists(source):
        os.symlink(source, target)


def main_logic(args: argparse.Namespace):
    # copy the regions and write comments
    regions = pd.read_csv(os.path.join(args.target_source, "regions.csv"), comment="#", header=None)
    to_csv_with_comments(regions, save_to="regions.csv",
                         additional_comment_lines=tuple([f"{k}: {v}" for k, v in vars(args).items()]),
                         write_directory_version=True, header=False, index=False)

    # create links for the simulated bulk signal
    pl_bulk = f"{args.experiment_name}.pl.bw"
    create_link(os.path.join(args.target_source, f"{args.target_name}.pl.bw"), pl_bulk)
    mn_bulk = f"{args.experiment_name}.mn.bw"
    create_link(os.path.join(args.target_source, f"{args.target_name}.mn.bw"), mn_bulk)
    # create links for the accessibility tracks
    sc_atac = []
    for sample in args.sample_labels:
        expected_atac = f"atac_ds.{sample}.bw"
        sc_atac.append(expected_atac)
        create_link(os.path.join(args.accessibility_source, expected_atac), expected_atac)
    scatac_norm_file = "scatac.norm.csv"
    # create links for scATAC norm factors
    create_link(os.path.join(args.accessibility_source, "scatac.meta.csv"), "scatac.meta.csv")
    create_link(os.path.join(args.accessibility_source, scatac_norm_file), scatac_norm_file)
    # create links for ground truth files
    ref_pls = []
    ref_mns = []
    for sample in args.sample_labels:
        expected_pl_file = os.path.join(args.target_source, f"{sample}.ds.pl.bw")
        if os.path.exists(expected_pl_file):
            create_link(expected_pl_file, f"{sample}.ds.pl.bw")
            ref_pls.append(f"{sample}.ds.pl.bw")
        expected_mn_file = os.path.join(args.target_source, f"{sample}.ds.mn.bw")
        if os.path.exists(expected_mn_file):
            create_link(expected_mn_file, f"{sample}.ds.mn.bw")
            ref_mns.append(f"{sample}.ds.mn.bw")

    prepare_dataset(**{
        "regions": "regions.csv",
        "final_regions": True,
        "bulk_pl": pl_bulk,
        "bulk_mn": mn_bulk,
        "target_sliding_sum": args.target_sliding_sum,
        "accessibility": sc_atac,
        "save_to": ".",
        "genome_fa": args.genome_fa,
        "chrom_size": args.chrom_size,
        "window_size": args.window_size,
        "ref_labels": args.sample_labels,
        "ref_pls": ref_pls,
        "ref_mns": ref_mns,
    })

    output_h5_file = "./data.h5"
    assert os.path.exists(output_h5_file) and os.path.exists(scatac_norm_file)
    with h5py.File(output_h5_file, "a") as fh:
        # write cluster labels
        gr = fh["dec"]
        gr.attrs["cluster_names"] = ",".join(args.sample_labels)

        norm = pd.read_csv(scatac_norm_file, header=None)
        transformed_norm = norm.copy()
        _mapping = {v: k for k, v in enumerate(norm[0].unique())}
        transformed_norm[0] = transformed_norm[0].map(_mapping)

        # save norm factors if any
        ds = fh.create_dataset("scatac_norm", data=transformed_norm.to_numpy())
        for cluster_str, cluster_idx in _mapping.items():
            ds.attrs[f"c_{cluster_idx}"] = cluster_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--experiment-name", type=str, required=True)
    parser.add_argument("--target-source", type=str, required=True,
                        help="Path to the folder containing the dataset for the bulk signal")
    parser.add_argument("--target-name", type=str, required=True,
                        help="Name of the target dataset")
    parser.add_argument("--accessibility-source", type=str, required=True,
                        help="Path to the folder containing the dataset for the accessibility signal")
    parser.add_argument("--sample-labels", nargs="+", type=str, required=True,)
    parser.add_argument("--target-sliding-sum", default=0, type=int,
                        help="Apply sliding sum to the target signals if the value is greater than 0.")
    parser.add_argument("-c", "--chrom-size", required=True)
    parser.add_argument("--genome-fa", type=str, required=True)
    parser.add_argument("--window-size", default=2_114, type=int)

    args = parser.parse_args()

    logger.info(args)

    main_logic(args)
