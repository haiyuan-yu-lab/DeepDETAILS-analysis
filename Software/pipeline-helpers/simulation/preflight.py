"""CLI entry: prepare scATAC fragments, call or load peaks, run DeepDETAILS preflight check.

Merges per-sample fragment files, optionally runs MACS2 (via tagAlign conversion), extends
peak summits, and calls :func:`deepdetails.helper.preflight.preflight_check`.
"""
from __future__ import annotations

import argparse
import inspect
import logging
import os
from typing import Any, Optional

import pandas as pd
from pythonase.run import run_command

try:
    from deepdetails.helper.preflight import preflight_check
except ImportError:
    from details.helper.preflight import preflight_check  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO)

_MIN_FRAGMENT_COLUMNS = 4
_MIN_PEAK_COLUMNS = 3


def _validate_fragment_table(df: pd.DataFrame, path: str) -> None:
    if df.shape[1] < _MIN_FRAGMENT_COLUMNS:
        raise ValueError(
            f"Fragment file {path!r} needs at least {_MIN_FRAGMENT_COLUMNS} columns "
            f"(chrom, start, end, barcode); got {df.shape[1]}."
        )
    if df.empty:
        raise ValueError(f"Fragment file {path!r} has no rows.")


def _validate_peak_table(df: pd.DataFrame, path: str) -> None:
    if df.shape[1] < _MIN_PEAK_COLUMNS:
        raise ValueError(
            f"Peak file {path!r} needs at least {_MIN_PEAK_COLUMNS} BED columns; got {df.shape[1]}."
        )
    if df.empty:
        raise ValueError(f"Peak file {path!r} has no rows.")


def call_macs2_for_atac(
    ta: str,
    gensz: str,
    prefix: str,
    pval_thresh: Optional[float],
    qval_thresh: Optional[float],
    macs2_exe: str = "macs2",
) -> tuple[str, ...]:
    """Call peaks with MACS2 on a fragment or BED-like file (converted to tagAlign in CWD).

    Parameters
    ----------
    ta : str
        tagAlign file for the treatment group
    gensz : str
        effective genome size for MACS2
    prefix : str
        Prefix for all output files
    pval_thresh : Optional[float]
        p-value cutoff. By default, use q-value cutoff as specified by the qval_thresh.
        However, the default behavior will be overridden if this parameter is given a float value.
    qval_thresh : Optional[float]
        q-value cutoff. One must specify either pval_thresh or qval_thresh
    macs2_exe : str
        Path to the MACS2 executable file

    Returns
    -------
    peak_files : tuple[str]
        Path to the peak file
    """
    # Convert fragments to ENCODE tagAlign as in
    # https://abc-enhancer-gene-prediction.readthedocs.io/en/latest/usage/scATAC.html
    cat_exe = "cat" if ta.find(".gz") == -1 else "zcat"
    to_tagalign = f"LC_ALL=C {cat_exe} {ta} | "
    to_tagalign += "sed '/^#/d' | "
    to_tagalign += "awk -v OFS='\\t' '{mid=int(($2+$3)/2); print $1,$2,mid,\"N\",1000,\"+\"; print $1,mid+1,$3,\"N\",1000,\"-\"}' | "
    to_tagalign += "sort -k 1,1V -k 2,2n -k3,3n --parallel 5 | bgzip -c > tagAlign.gz"
    logger.debug("tagAlign pipeline: %s", to_tagalign)
    _stdout, stderr, rc = run_command(to_tagalign)
    if rc != 0:
        raise RuntimeError(stderr or f"tagAlign conversion failed with exit code {rc}")

    model_str = f"--nomodel --shift -75 --extsize 150"
    if pval_thresh:
        cutoff = f"-p {pval_thresh}"
    elif qval_thresh:
        cutoff = f"-q {qval_thresh}"
    else:
        raise ValueError("You must specify either a p-value or q-value cutoff.")

    _stdout, stderr, rc = run_command(
        f"{macs2_exe} callpeak "
        f"-t tagAlign.gz -f BED -n {prefix} -g {gensz} {cutoff} "
        f"{model_str} --keep-dup all"
    )
    if rc != 0:
        raise RuntimeError(stderr or f"MACS2 callpeak failed with exit code {rc}")
    expected_peak_file = f"{prefix}_peaks.narrowPeak"
    if not os.path.exists(expected_peak_file):
        raise OSError(f"Expected peak file missing after MACS2: {expected_peak_file}")

    return (expected_peak_file,)


def _preflight_extra_kwargs(sig_prior: Optional[str]) -> dict[str, Any]:
    """Pass ``sig_prior`` only if the installed ``preflight_check`` accepts it."""
    params = inspect.signature(preflight_check).parameters
    if sig_prior is not None and "sig_prior" in params:
        return {"sig_prior": sig_prior}
    if sig_prior is not None:
        logger.warning(
            "prior.bed was written but this install's preflight_check has no sig_prior "
            "argument; run a matching library version or use the file downstream manually."
        )
    return {}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare fragments and peaks, then run the DeepDETAILS preflight check.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preflight-cutoff",
        default=0.5,
        type=float,
        help="Detection cutoff passed to preflight_check (frac-based diagnosis).",
    )
    parser.add_argument("--sample-labels", nargs="+", type=str, required=True, help="Labels per --raw-frags sample.")
    parser.add_argument("--bulk-pl", type=str, required=True, help="Bulk plus-strand BigWig path.")
    parser.add_argument("--bulk-mn", type=str, default=None, help="Bulk minus-strand BigWig (optional).")
    parser.add_argument("--raw-frags", nargs="+", type=str, required=True, help="Fragment TSV paths (10x-like).")
    parser.add_argument(
        "--acc-peaks",
        type=str,
        default=None,
        help="If set and the file exists, skip MACS2 and use this narrowPeak/BED.",
    )
    parser.add_argument("--save-to", default=".", type=str, help="Directory for intermediates and outputs.")
    parser.add_argument("--genome-size", default="hs", type=str, help="MACS2 -g effective genome size or shorthand.")
    parser.add_argument("--macs2-exec", default="macs2", type=str, help="MACS2 executable.")
    parser.add_argument("--macs2-qval", default=0.01, type=float, help="MACS2 q-value when not using -p.")
    parser.add_argument(
        "--atac-extension",
        default=500,
        type=int,
        help="Half-width (bp) around peak summit for region definition.",
    )
    parser.add_argument("--top-n", default=1000, type=int, help="max_top_n for preflight_check.")
    parser.add_argument(
        "--sig-folder",
        type=str,
        default=None,
        help="Folder with per-sample signature tables ``{label}.txt`` (optional).",
    )
    parser.add_argument(
        "--promoter-bed",
        type=str,
        default=None,
        help="Promoter BED with Ensembl_ID column (optional; used with --sig-folder).",
    )

    return parser.parse_args()


def main() -> None:
    """Load fragments, build barcodes, define peaks, optionally compile prior.bed, run preflight."""
    args = get_args()

    if len(args.sample_labels) != len(args.raw_frags):
        raise ValueError(
            f"--sample-labels ({len(args.sample_labels)}) and --raw-frags ({len(args.raw_frags)}) "
            "must have the same length."
        )

    os.makedirs(args.save_to, exist_ok=True)

    frag_dfs: list[pd.DataFrame] = []
    barcode_dfs: list[pd.DataFrame] = []
    frag_dict: dict[str, str] = {}

    for ct, frag_file in zip(args.sample_labels, args.raw_frags):
        logger.info("Preparing fragments for %s (%s)...", ct, frag_file)
        _frag_df = pd.read_csv(frag_file, sep="\t", header=None)
        _validate_fragment_table(_frag_df, frag_file)
        _frag_df = _frag_df.copy()
        _frag_df[3] = _frag_df[3].astype(str).str.replace("-1", f"-{ct}", regex=False)
        tmp_file = os.path.join(args.save_to, f"{ct}.fragments.tsv")
        frag_dict[ct] = tmp_file
        _frag_df.sort_values([0, 1]).to_csv(tmp_file, sep="\t", header=False, index=False)

        frag_dfs.append(_frag_df)
        logger.info("Extracting barcodes for %s...", ct)
        barcode_dfs.append(pd.DataFrame({0: _frag_df[3].unique(), 1: ct}))

    barcode_file = os.path.join(args.save_to, "barcodes.tsv")
    barcodes = pd.concat(barcode_dfs, ignore_index=True)
    barcodes.to_csv(barcode_file, sep="\t", header=False, index=False)

    if args.acc_peaks is not None and os.path.isfile(args.acc_peaks):
        logger.info("Using existing accessibility peaks (%s); skipping MACS2.", args.acc_peaks)
        outs = [args.acc_peaks]
    else:
        if args.acc_peaks is not None:
            logger.warning("acc-peaks path does not exist (%s); running MACS2 instead.", args.acc_peaks)
        merged_fragment_file = os.path.join(args.save_to, "merged.fragments.tsv")
        logger.info("Exporting merged fragments to %s", merged_fragment_file)
        pd.concat(frag_dfs, ignore_index=True).to_csv(
            merged_fragment_file, sep="\t", header=False, index=False
        )

        logger.info("Calling accessible regions with MACS2...")
        outs = list(
            call_macs2_for_atac(
                merged_fragment_file,
                args.genome_size,
                "atac",
                pval_thresh=None,
                qval_thresh=args.macs2_qval,
                macs2_exe=args.macs2_exec,
            )
        )
    del frag_dfs

    peaks = pd.read_csv(outs[0], sep="\t", comment="#", header=None)
    _validate_peak_table(peaks, outs[0])
    mids = (0.5 * (peaks[1] + peaks[2])).astype(int)
    peaks = peaks.copy()
    peaks[1] = mids - args.atac_extension
    peaks[2] = mids + args.atac_extension
    peaks = peaks.loc[(peaks[1] >= 0) & (peaks[0].astype(str).str.startswith("chr")), :].copy()
    peaks = peaks.reset_index(drop=True)

    sig_prior: Optional[str] = None
    sig_folder = args.sig_folder or ""
    promoter_bed = args.promoter_bed or ""
    if sig_folder and os.path.isdir(sig_folder) and promoter_bed and os.path.isfile(promoter_bed):
        sig_paths = [os.path.join(sig_folder, f"{c}.txt") for c in args.sample_labels]
        if all(os.path.isfile(p) for p in sig_paths):
            sig_dfs = [pd.read_csv(p, sep="\t") for p in sig_paths]
            sigs = pd.concat(sig_dfs, ignore_index=True)
            if "gene_id" not in sigs.columns:
                raise ValueError(
                    f"Signature tables under {sig_folder!r} must include a 'gene_id' column; got {list(sigs.columns)}."
                )
            sigs["major"] = sigs["gene_id"].astype(str).str.split(".", expand=True)[0]

            promoters = pd.read_csv(promoter_bed, sep="\t")
            id_col = "Ensembl_ID"
            if id_col not in promoters.columns:
                raise ValueError(
                    f"Promoter BED {promoter_bed!r} must include an {id_col!r} column; got {list(promoters.columns)}."
                )
            sig_prior = os.path.join(args.save_to, "prior.bed")
            sig_prior_df = promoters.loc[promoters[id_col].astype(str).isin(sigs["major"])]
            logger.info("Compiled %d prior signature regions -> %s", sig_prior_df.shape[0], sig_prior)
            sig_prior_df.to_csv(sig_prior, sep="\t", header=False, index=False)
        else:
            missing = [p for p in sig_paths if not os.path.isfile(p)]
            logger.warning("Skipping prior.bed: missing signature files: %s", missing)
    elif sig_folder or promoter_bed:
        logger.warning(
            "Both --sig-folder (existing directory) and --promoter-bed (existing file) are required for prior.bed; skipping."
        )

    logger.info("Running preflight check")
    bulks: tuple[Optional[str], Optional[str]] = (args.bulk_pl, args.bulk_mn)
    extra = _preflight_extra_kwargs(sig_prior)
    preflight_check(
        frag_dict,
        barcodes,
        peaks,
        bulks,
        preflight_cutoff=args.preflight_cutoff,
        max_top_n=args.top_n,
        **extra,
    )


if __name__ == "__main__":
    main()
