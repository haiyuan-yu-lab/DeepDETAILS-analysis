#!/usr/bin/env python
import argparse
import gzip
import h5py
import logging
import os
import pybedtools
import pyBigWig
import numpy as np
import pandas as pd
from multiprocessing import Pool
from typing import Union, Optional
from pythonase.run import run_command
from downsample_bedlike import downsample_bed
from downsample_bws import downsample_bw
from downsample_sc_libs import downsample_sclib, frag_file_to_bw
from utils import bed_to_cov_bw
from pints.calling_engine import peak_calling
from pints.extension_engine import extend
from deepdetails.protocols import prepare_dataset


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO)


def get_total_counts(files: Union[list[str], tuple[str, ...]], input_type: str = "end_bw") -> Union[int, float]:
    """Get total read counts in the pl-mn bigwig file pairs

    Parameters
    ----------
    files : Union[list[str], tuple[str, ...]]
        Path to the file(s) that should be counted
    input_type : str, optional
        Signal type in the bigWig files, by default "end_bw"

    Returns
    -------
    Union[int, float]
        counts

    Raises
    ------
    IOError
        If any input file cannot be located
    ValueError
        If the input_type is not supported
    """

    def _get_bw_counts(file_path):
        if os.path.exists(file_path):
            with pyBigWig.open(file_path) as bw_obj:
                return np.abs(bw_obj.header()["sumData"])
        else:
            raise IOError(f"Cannot find {file_path}")

    counts = 0
    for f in files:
        if input_type == "end_bw":
            counts += _get_bw_counts(f)
        elif input_type == "tagAlign":
            compression = pd.io.common.infer_compression(f, "infer")
            if compression is None:
                with open(f, "r") as f:
                    for _ in f:
                        counts += 1
            elif compression == "gzip":
                with gzip.open(f) as f:
                    for _ in f:
                        counts += 1
            else:
                raise IOError(f"Unsupported compression method: {compression}")
        elif input_type == "counts":
            # expect a tsv file with header
            _df = pd.read_csv(f, sep="\t")
            assert "expected_count" in _df.columns
            counts = _df["expected_count"].sum()
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
    return counts


def get_sampling_ratios(sample_labels: Union[list[str], tuple[str, ...]],
                        contrib_ratio: Union[list[float], tuple[float, ...]],
                        input_files: dict[str, Union[list[str], tuple[str, ...]]],
                        total_counts: int, pe_adjustment: Union[list[int], tuple[int, ...]] = 1,
                        raw_format: str = "end_bw") -> tuple[dict, dict]:
    """Get sampling ratios for each lib

    Parameters
    ----------
    sample_labels : Union[list[str], tuple[str, ...]]
        Sample labels
    contrib_ratio : Union[list[float], tuple[float, ...]]
        Contribution rate for each sample
    input_files : dict[str, Union[list[str], tuple[str, ...]]]
        key: sample label
        value: files for the sample
    total_counts : int
        Total number of reads counts to be sampled. Set as 0, if you want to keep all counts.
    pe_adjustment : Union[list[int], tuple[int, ...]]
        Adjustment coefficient for each library.
        1 for no extra adjustment, 2 for double the amount of samples
    raw_format : str
        end_bw for bigWig input
        tagAlign for ENCODE tagAlign
        frags for read fragments (not yet)

    Returns
    -------
    norm_contrib : dict
        key: sample
        value: normalized contrib ratios
    result : dict
        key: sample
        value: sampling ratio/count
    """
    contrib = {sl: sr / sum(contrib_ratio) for sl, sr in zip(sample_labels, contrib_ratio)}
    per_sample_counts = {}
    sampling_ratios = {}
    sampling_counts = {}
    if total_counts > 0:
        for idx, s in enumerate(sample_labels):
            sampling_counts[s] = int(contrib[s] * total_counts * pe_adjustment[idx])
            per_sample_counts[s] = get_total_counts(input_files[s], input_type=raw_format)
            sampling_ratios[s] = sampling_counts[s] / per_sample_counts[s]
            logger.info(
                f"Sample {s} has {per_sample_counts[s]} counts, and it takes up {contrib[s]:.4%} counts in the final lib. "
                f"The effective sampling ratio will be {sampling_ratios[s]:.4f}.")
            if sampling_ratios[s] > 1.:
                logger.warning(f"Sampling ratio for {s} is greater than 1!")
    else:
        logger.warning("total_counts is specified as 0, setting sampling ratios to 1.")
        sampling_ratios = {k: 1. for k in sample_labels}
    return contrib, sampling_ratios


def _tagalign_ds_mult_thread_core(
        _sample_files: Union[list[str], tuple[str, ...]],
        _sampling_ratio: float, _sample_label: str,
        _random_generator: np.random._generator.Generator,
        _chrom_size: str, _suffix: str = "", _full_cov: bool = False) -> tuple[str, str, str]:
    """Single-thread downsampling task for tagAlign inputs

    Parameters
    ----------
    _sample_files : Union[list[str], tuple[str, ...]]
        tagAlign files
    _sampling_ratio : float
        Probability of keeping a tag
    _sample_label : str
        Sample label, this will be used as the prefix for all outputs
    _random_generator : np.random._generator.Generator
        Random number generator
    _chrom_size : str
        Path to the chromosome size file
    _suffix : str
        Suffix for the outputs
    _full_cov : bool
        Set as True if you want to use the entire read for coverage calculation

    Returns
    -------
    tuple[str, str, str]
        Path to the downsampled tagAlign file
        Path to the forward strand signal file (derived from the downsampled tagAlign file)
        Path to the reverse strand signal file (derived from the downsampled tagAlign file)
    """
    logger.info(f"Downsampling {_sample_files} with keeping prob {_sampling_ratio}")
    _ds_frag_file = downsample_bed(
        _sample_files[0], _sampling_ratio,
        save_to=f"{_sample_label}{_suffix}.ds.tagAlign.gz", random_generator=_random_generator, scope=None)
    _ds_pl_file = f"{_sample_label}{_suffix}.ds.pl.bw"
    bed_to_cov_bw(
        in_bed_path=_ds_frag_file,
        out_bigwig_path=_ds_pl_file,
        chrom_size_path=_chrom_size, report_5p_cov=not _full_cov, limit_strand_to="+"
    )
    _ds_mn_file = f"{_sample_label}{_suffix}.ds.mn.bw"
    bed_to_cov_bw(
        in_bed_path=_ds_frag_file,
        out_bigwig_path=_ds_mn_file,
        chrom_size_path=_chrom_size, report_5p_cov=not _full_cov, limit_strand_to="-"
    )
    return _ds_frag_file, _ds_pl_file, _ds_mn_file


def run_down_sampling(sample_labels: Union[list[str], tuple[str, ...]],
                      input_files: dict[str, Union[list[str], tuple[str, ...]]],
                      sampling_ratios: dict[str, float], chrom_size: str,
                      random_generator: np.random._generator.Generator, use_poisson: bool = False,
                      raw_format: str = "end_bw", threads: int = 1,
                      suffix: str = "") -> dict:
    """Downsample libraries

    Parameters
    ----------
    sample_labels : Union[list[str], tuple[str, ...]]

    input_files : dict[str, Union[list[str], tuple[str, ...]]]

    sampling_ratios: dict[str, float]

    chrom_size : str
        Path to the chromosome size file
    random_generator : np.random_generator.Generator

    use_poisson : bool
        Use a Poisson sampler instead of the default Binomial sampler
    raw_format : str
        Format of the input files.
    threads : int
        Threads for downsampling
    suffix : str
        Suffix for the outputs

    Returns
    -------
    dict
        key: pl, mn
        values: dict
            key: sample
            value: file path
    """
    if raw_format == "end_bw":
        _downsampled_files = {"pl": {}, "mn": {}}
        for i, s in enumerate(sample_labels):
            sample_files = input_files[s]
            logger.info(f"Downsampling {sample_files} with keeping prob {sampling_ratios[s]}")
            plf = downsample_bw(
                sample_files[0], output_prefix=f"{s}{suffix}.ds.pl",
                chrom_size=chrom_size, keep_prob=sampling_ratios[s],
                random_generator=random_generator, use_poisson=use_poisson)
            _downsampled_files["pl"][s] = plf
            if len(sample_files) == 2:
                mnf = downsample_bw(
                    sample_files[1], output_prefix=f"{s}{suffix}.ds.mn",
                    chrom_size=chrom_size, keep_prob=sampling_ratios[s],
                    random_generator=random_generator, use_poisson=use_poisson)
                _downsampled_files["mn"][s] = mnf
            elif len(sample_files) > 2:
                raise ValueError("sample_files should have at most 2 files")
    elif raw_format.find("tagAlign") != -1:  # tagAlign or control-tagAlign
        _downsampled_files = {"ds-frags": {}, "pl": {}, "mn": {}}
        if use_poisson:
            raise NotImplementedError("Poisson sampler for tagAlign inputs have not been implemented yet.")
        jobs = []
        use_full = True if raw_format.find("full") != -1 else False
        for s in sample_labels:
            sample_files = input_files[s]
            jobs.append((sample_files, sampling_ratios[s], s, random_generator, chrom_size, suffix, use_full))
        with Pool(threads) as pool:
            results = pool.starmap(_tagalign_ds_mult_thread_core, jobs)
        for (s, r) in zip(sample_labels, results):
            _downsampled_files["ds-frags"][s] = r[0]
            _downsampled_files["pl"][s] = r[1]
            _downsampled_files["mn"][s] = r[2]
    else:
        raise ValueError(f"raw_format {raw_format} not supported")

    return _downsampled_files


def simulate_bulk_tracks(files: Union[tuple, list], chrom_sizes: str, prefix: str = "",
                         scale: Union[int, float] = 1, to_int: bool = True,
                         wiggle_exe: str = "wiggletools", wig2bw_exe: str = "wigToBigWig") -> str:
    """Simulate bulk tracks

    Parameters
    ----------
    files : Union[tuple,list]
        Downsampled bigWig files to be merged
    chrom_sizes : str

    prefix : str, optional
        by default ""
    scale : Union[int, float], optional
        Constant value that will be applied to all signal values, by default 1 (keep the original values)
    to_int : bool, optional
        Convert signal values to integers, by default True
    wiggle_exe : str, optional
        Path to the wiggletools executable, by default "wiggletools"
    wig2bw_exe : str, optional
        Path to the wigToBigWig executable, by default "wigToBigWig"

    Returns
    -------
    str
        Path to the merged file
    """
    dest_merged_wig = f"{prefix}.merged.wig"
    dest_merged_bw = f"{prefix}.bw"
    cmd = f"{wiggle_exe} sum {' '.join(files)} | {wiggle_exe} scale {scale} - > {dest_merged_wig}"
    logger.info(cmd)

    run_command(cmd, raise_exception=True)
    if to_int:
        cmd = f"{wiggle_exe} toInt {dest_merged_wig} > {dest_merged_wig}.1 && mv {dest_merged_wig}.1 {dest_merged_wig}"
        logger.info(cmd)
        run_command(cmd, raise_exception=True)
    cmd = f"{wig2bw_exe} {dest_merged_wig} {chrom_sizes} {dest_merged_bw}"
    logger.info(cmd)
    run_command(cmd, raise_exception=True)
    os.remove(dest_merged_wig)
    return dest_merged_bw


def simulate_bulk_tags(files: Union[tuple[str, ...], list[str]], prefix: str) -> str:
    """Simulate bulk lib from downsampled tagAligns

    Parameters
    ----------
    files : Union[tuple[str, ...], list[str]]
        Downsampled tagAlign files to be merged
    prefix : str
        Output prefix

    Returns
    -------
    str
        Path to the merged tagAlign file
    """
    dest_merged_ta = f"{prefix}.merged.tagAlign"
    cmd = f"zcat {' '.join(files)} > {dest_merged_ta}"
    logger.info(cmd)

    run_command(cmd, raise_exception=True)
    return dest_merged_ta


def call_pints(pl_bulk: str, mn_bulk: str, output_prefix: str,
               fdr_target: float, threads: int) -> tuple[str, str]:
    """Call peaks with PINTS (for tss-assays)

    Parameters
    ----------
    pl_bulk : str
        Path to the forward signal track
    mn_bulk : str
        Path to the reverse signal track
    output_prefix : str
        Prefix for all output files
    fdr_target : float
        FDR target
    threads : int
        Number of threads for peak calling

    Returns
    -------
    tuple[str, str]
        Paths to bidirectional and unidirectional calls.
    """
    logger.info("Start peak calling...")
    peak_calling(input_bam=None, bw_pl=(pl_bulk,), bw_mn=(mn_bulk,),
                 output_dir=".", output_prefix=output_prefix,
                 fdr_target=fdr_target, thread_n=threads)
    expected_bid_file = f"{output_prefix}_1_bidirectional_peaks.bed"
    expected_div_file = f"{output_prefix}_1_divergent_peaks.bed"
    expected_uni_file = f"{output_prefix}_1_unidirectional_peaks.bed"
    if not all(map(
            lambda x: os.path.exists(x),
            (expected_bid_file, expected_div_file, expected_uni_file))):
        logger.exception("Missing expected outputs from PINTS")
    ext_args = {
        "bam_files": None,
        "bw_pl": (pl_bulk,),
        "bw_mn": (mn_bulk,),
        "divergent_files": (expected_div_file,),
        "bidirectional_files": (expected_bid_file,),
        "unidirectional_files": (expected_uni_file,),
        "save_to": ".",
        "div_ext_left": (60,),
        "div_ext_right": (60,),
        "unidirectional_ext_left": (60,),
        "unidirectional_ext_right": (60,),
        "promoter_bed": None,
    }
    # pints sets a tmp dir for pybedtools and that dir will be
    # deleted after peak calling, in the following, we need to reset
    # the tmp dir to an accessible place
    logger.info("Start peak extension")
    pybedtools.set_tempdir(".")
    extend(argparse.Namespace(**ext_args))
    expected_bid_file = f"{output_prefix}_1_bidirectional_peaks_element_60bp.bed"
    expected_uni_file = f"{output_prefix}_1_unidirectional_peaks_element_60bp.bed"
    files = (expected_bid_file, expected_uni_file)
    if not all(map(lambda x: os.path.exists(x), files)):
        logger.exception("Missing expected outputs from PINTS extension")
    return files


def call_macs2(ta: str, gensz: str, prefix: str, pval_thresh: Optional[float],
               qval_thresh: Optional[float], ctl_ta: Optional[str] = None,
               frag_len: Optional[int] = None, macs2_exe: str = "macs2") -> tuple[str]:
    """Call peaks with MACS2 (for histone modification ChIP-seq)

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
    ctl_ta : Optional[str]
        tagAlign file for the control group
    frag_len : Optional[int]
        Fragment length as estimated by cross-correlation analysis
    macs2_exe : str
        Path to the MACS2 executable file

    Returns
    -------
    peak_files : tuple[str]
        Path to the peak file
    """
    if frag_len:
        model_str = f"--nomodel --shift 0 --extsize {frag_len}"
    else:
        model_str = ""
    if pval_thresh:
        cutoff = f"-p {pval_thresh}"
    elif qval_thresh:
        cutoff = f"-q {qval_thresh}"
    else:
        raise ValueError("You must specify either a p-value or q-value cutoff.")
    ctl_param = '-c {ctl_ta}'.format(ctl_ta=ctl_ta) if ctl_ta else ''
    stdout, stderr, rc = run_command(
        f"{macs2_exe} callpeak "
        f"-t {ta} {ctl_param} -f BED -n {prefix} -g {gensz} {cutoff} "
        f"{model_str} --keep-dup all -B --SPMR"
    )
    if rc != 0:
        raise RuntimeError(stderr)
    expected_peak_file = f"{prefix}_peaks.narrowPeak"
    if not os.path.exists(expected_peak_file):
        raise IOError(f"Cannot locate the expected peak file {expected_peak_file}")

    return (expected_peak_file,)


def downsample_scatac_libs(total_cells: int, cell_ratios: dict, sample_labels: tuple, sc_atac: tuple,
                           mean_frags_per_cell: tuple, std_frags_per_cell: tuple,
                           cell_barcodes: Union[tuple, None], chrom_size: str, threads: int = 1,
                           random_generator=None, frag_proc: str = "naive", rpm_norm: bool = False) -> tuple[tuple, pd.DataFrame]:
    """Downsample sc/snATAC-seq libraries

    Parameters
    ----------
    total_cells : int
        Sample how many cells?
    cell_ratios : dict
        _description_
    sample_labels : tuple
        _description_
    sc_atac : tuple
        _description_
    mean_frags_per_cell : tuple
        _description_
    std_frags_per_cell : tuple
        _description_
    cell_barcodes : Union[tuple, None]
        _description_
    chrom_size : str
        _description_
    threads : int, optional
        _description_, by default 1
    random_generator : _type_, optional
        _description_, by default None
    frag_proc : str, optional
        _description_, by default "naive"
    rpm_norm : bool, optional
        by default False

    Returns
    -------
    expected_downsampled_files : tuple
        _description_
    sample_info : pd.DataFrame
        Downsampled info
    """
    if len(mean_frags_per_cell) == 1:  # broadcasting
        mean_frags_per_cell = mean_frags_per_cell * len(sc_atac)
        std_frags_per_cell = std_frags_per_cell * len(sc_atac)
    if cell_barcodes is None or (isinstance(cell_barcodes, list) and len(cell_barcodes) == 0):
        cell_barcodes = [None, ] * len(sc_atac)
    atac_ds_jobs = []
    n_cells_per_sample = {k: int(v * total_cells) for k, v in cell_ratios.items()}
    expected_downsampled_files = []
    for i, (s, frag_file) in enumerate(zip(sample_labels, sc_atac)):
        logger.info(f"Downsampling {n_cells_per_sample[s]} cells from "
                    f"sc/snATAC-lib {frag_file} with mean frags per cell {mean_frags_per_cell[i]} "
                    f"std frags per cell {std_frags_per_cell[i]}.")
        dest = f"atac_ds.{s}.tsv.gz"
        logger.info(f" - Downsampled file will be written to {dest}.")
        atac_ds_jobs.append((
            frag_file, n_cells_per_sample[s], mean_frags_per_cell[i],
            std_frags_per_cell[i], dest, cell_barcodes[i],
            chrom_size, random_generator, frag_proc, rpm_norm))
        expected_downsampled_files.append(f"atac_ds.{s}.bw")

    with Pool(threads) as p:
        sampled_lib_sizes = p.starmap(downsample_sclib, atac_ds_jobs)
    samples = []
    for meta, lib_size in zip(atac_ds_jobs, sampled_lib_sizes):
        samples.append((meta[4], lib_size))
    return tuple(expected_downsampled_files), pd.DataFrame(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--experiment-name", type=str, required=True)

    group = parser.add_argument_group("About the samples")
    group.add_argument("--sample-labels", nargs="+", type=str, required=True, default=("a",))
    in_group = group.add_mutually_exclusive_group(required=True)
    in_group.add_argument("--raw-pl", nargs="+")
    in_group.add_argument("--raw-frags", nargs="+")
    in_group.add_argument("--raw-counts", nargs="+", help="Counts table for each cluster")
    group.add_argument("--raw-mn", nargs="+")
    group.add_argument("--raw-control-frags", nargs="*")
    group.add_argument("--target-sliding-sum", default=0, type=int,
                       help="Apply sliding sum to the target signals if the value is greater than 0.")
    group.add_argument("--sc-atac", nargs="+", required=True)
    group.add_argument("--raw-format", choices=("end_bw", "tagAlign", "tagAlign-full"),
                       default="end_bw")
    group.add_argument("--paired-end-adjust", nargs="+", type=int, default=0, choices=(0, 1),
                       help="For signals derived from paired-end tagAlign files, "
                            "you may need to set value as 1 for the corresponding files")
    group.add_argument("--paired-end-adjust-control", nargs="+", type=int, default=0, choices=(0, 1),
                       help="Similar to --paired-end-adjust, this is the option for control signals")
    group.add_argument("--prior", help="Priors on similarities between clusters. "
                                       "Row and column names must match --sample-labels")

    group = parser.add_argument_group("About the sampling experiment")
    group.add_argument("--contrib-ratio", nargs="+", default=1., type=float)
    group.add_argument("--atac-contrib-ratio", nargs="+", type=float)
    group.add_argument("--total-counts", type=int, default=31_000_000)
    group.add_argument("--total-control-counts", type=int)
    group.add_argument("--total-cells", type=int)
    group.add_argument("-r", "--random-seed", type=int, default=1234567)

    group = parser.add_argument_group("Peak calling")
    group.add_argument("--peaks", nargs="+", type=str, required=False,
                       help="Predefined peaks. If provided, this script will not call peaks.")
    group.add_argument("-f", "--final-peaks", action="store_true",
                       help="If specified, the provided peaks are final and will be used for building the dataset.")
    group.add_argument("--merge-overlap-peaks", type=int, default=0,
                       help="Minimum overlap between features allowed for features to be merged. "
                            "Default is not to merge (0).")
    group.add_argument("-e", "--engine", choices=("PINTS", "MACS2"), default="PINTS")
    group.add_argument("--fdr-target", type=float, default=0.3)
    group.add_argument("--threads", type=int, default=16)
    group.add_argument("--macs2-exec", type=str, default="macs2", help="Path to the MACS2 executable file")
    group.add_argument("--macs2-gs", type=str, default="hs")
    group.add_argument("--macs2-fraglen", type=int, nargs="+",
                       help="Fragment length for each sample library obtained from cross-correlation analysis")
    group.add_argument("--macs2-pval", type=float,
                       help="If set, this will be the p-value cutoff for all MACS2 peaks. "
                            "Otherwise, q-value cutoff, as specified by --fdr-target, will be applied."
                            "Empirically, p-val cutoff should be used if you want peak calling to be less stringent (0.01).")

    group = parser.add_argument_group("sc/snATAC-seq downsampling")
    group.add_argument("-m", "--mean-frags-per-cell", nargs="+", type=int,
                       help="Average number of fragments to be sampled from each cell. "
                            "Should be either a single value or a list of values for each library")
    group.add_argument("-s", "--std-frags-per-cell", nargs="+", type=float,
                       help="Standard deviation of fragments to be sampled from each cell."
                            "Should be either a single value or a list of values for each library")
    group.add_argument("--cell-barcodes", nargs="*",
                       help="Allowed cell barcodes for each library (barcodes that passing filters)."
                            "All barcodes will be allowed if nothing is specified.")
    group.add_argument("--frag-processing", type=str, required=False,
                       choices=("naive", "encode", "cellranger", "5pi"), )
    group.add_argument("--skip-downsampling", action="store_true", help="Skip downsampling sc/snATAC-seq libs")
    group = parser.add_argument_group("PyTorch Dataset")
    group.add_argument("--background-sampling-ratio", default=0., type=float)
    group.add_argument("--background-blacklist", type=str, required=False)
    group.add_argument("--foreground-blacklist", type=str, required=False)
    group.add_argument("--genome-fa", type=str, required=True)
    group.add_argument("--window-size", default=4_096, type=int)
    group.add_argument("--skip-hdf5", dest="build_hdf5", action="store_false",
                       help="Downsample and simulate the bulk/reference files only, skip building the HDF5 file")
    group.add_argument("--rpm-norm", action="store_true", help="Use RPM to normalize ATAC signal")

    group = parser.add_argument_group("Dependencies")
    group.add_argument("-c", "--chrom-size", required=True)
    group.add_argument("--wiggletools", default="wiggletools")
    group.add_argument("--wig2bw", default="wigToBigWig")

    args = parser.parse_args()
    if args.raw_mn and not args.raw_pl:
        parser.error("--raw-mn is allowed only when --raw-pl is specified")
    if args.raw_control_frags and not args.raw_frags:
        parser.error("--raw-control-frags is allowed only when --raw-frags is specified")
    if args.total_control_counts and not args.raw_control_frags:
        logger.warning(f"--total-control-counts is specified as {args.total_control_counts} but no control fragments were provided.")
        logger.warning("The option will not be effective in downstream analysis.")
    if not args.skip_downsampling:
        required_opts = {"total_cells": args.total_cells, "sample_labels": args.sample_labels,
                         "sc_atac": args.sc_atac, "mean_frags_per_cell": args.mean_frags_per_cell,
                         "std_frags_per_cell": args.std_frags_per_cell,
                         "cell_barcodes": args.cell_barcodes, "frag_processing": args.frag_processing}
        _missing_opts = [k for k, v in required_opts.items() if v is None]
        if len(_missing_opts) > 0:
            parser.error(f"Missing value(s) for opt(s) {_missing_opts}")

    if args.final_peaks and len(args.peaks) > 1:
        parser.error("When --final-peaks is specified, you can only provide one peak file.")

    logger.info(args)

    n_samples = len(args.sample_labels)
    n_input_files = 0
    if args.raw_pl:
        n_input_files = len(args.raw_pl)
    else:
        n_input_files = len(args.raw_frags)
    if isinstance(args.contrib_ratio, int) or isinstance(args.contrib_ratio, float):
        args.contrib_ratio = [args.contrib_ratio] * n_samples
    if len(args.contrib_ratio) == 1:
        args.contrib_ratio = [args.contrib_ratio[0]] * n_samples
    if isinstance(args.paired_end_adjust, int) or isinstance(args.paired_end_adjust, float):
        args.paired_end_adjust = [args.paired_end_adjust] * n_samples
    if len(args.paired_end_adjust) == 1:
        args.paired_end_adjust = [args.paired_end_adjust[0]] * n_samples
    args.paired_end_adjust = [a + 1 for a in args.paired_end_adjust]
    if isinstance(args.paired_end_adjust_control, int) or isinstance(args.paired_end_adjust_control, float):
        args.paired_end_adjust_control = [args.paired_end_adjust_control] * n_samples
    if len(args.paired_end_adjust_control) == 1:
        args.paired_end_adjust_control = [args.paired_end_adjust_control[0]] * n_samples
    args.paired_end_adjust_control = [a + 1 for a in args.paired_end_adjust_control]
    if args.atac_contrib_ratio is None:
        # if the contribution ratio for ATAC-seq libs are not specified, then use the same ratios for target libs
        args.atac_contrib_ratio = args.contrib_ratio
    elif isinstance(args.atac_contrib_ratio, int) or isinstance(args.atac_contrib_ratio, float):
        args.atac_contrib_ratio = [args.atac_contrib_ratio] * n_samples
    if len(args.atac_contrib_ratio) == 1:
        args.atac_contrib_ratio = [args.atac_contrib_ratio[0]] * n_samples

    if n_input_files != n_samples:
        parser.error("--raw-pl or --raw-frags must have the same length as --sample-labels")
    if n_samples != len(args.contrib_ratio):
        parser.error("--contrib-ratio must have the same length as --sample-labels")
    if n_samples != len(args.sc_atac):
        parser.error("--sc-atac must have the same length as --sample-labels")
    if args.raw_mn is not None and len(args.raw_mn) != n_samples:
        parser.error("If set, --raw-mn must have the same length as --raw-pl")
    if (args.raw_control_frags is not None and len(args.raw_control_frags) > 0) and len(args.raw_control_frags) != n_samples:
        parser.error("If set, --raw-control-frags must have the same length as --raw-frags")

    random_g = np.random.default_rng(args.random_seed)

    if args.raw_format == "end_bw":
        input_files = {
            s: (args.raw_pl[idx], args.raw_mn[idx]) if args.raw_mn[idx] else (args.raw_pl[idx],) for idx, s in
            enumerate(args.sample_labels)}
        # get downsample ratios for each sample
        contrib_ratios, sampling_ratios = get_sampling_ratios(
            args.sample_labels, args.contrib_ratio, input_files=input_files,
            total_counts=args.total_counts, pe_adjustment=args.paired_end_adjust)
        # run downsampling on the target signal
        downsampled_files = run_down_sampling(
            sample_labels=args.sample_labels, input_files=input_files,
            sampling_ratios=sampling_ratios, chrom_size=args.chrom_size, random_generator=random_g, raw_format="end_bw"
        )
        # simulate bulk tracks
        pl_bulk = simulate_bulk_tracks(
            files=downsampled_files["pl"].values(),
            chrom_sizes=args.chrom_size,
            prefix=f"{args.experiment_name}.pl", scale=1,
            to_int=True, wiggle_exe=args.wiggletools,
            wig2bw_exe=args.wig2bw
        )
        mn_bulk = simulate_bulk_tracks(
            files=downsampled_files["mn"].values(),
            chrom_sizes=args.chrom_size,
            prefix=f"{args.experiment_name}.mn", scale=1,
            to_int=True, wiggle_exe=args.wiggletools,
            wig2bw_exe=args.wig2bw
        )
        # call peaks
        if args.peaks is None:
            peaks = call_pints(pl_bulk, mn_bulk, args.experiment_name, args.fdr_target, args.threads)
            if args.foreground_blacklist is not None:
                for f in peaks: pybedtools.BedTool(f).intersect(
                    pybedtools.BedTool(args.foreground_blacklist), v=True).saveas(f)
        else:
            peaks = args.peaks
    elif args.raw_format == "tagAlign" or args.raw_format == "tagAlign-full":
        # get downsample ratios for each sample
        exp_files = {
            s: (args.raw_frags[idx],) for idx, s in enumerate(args.sample_labels)}
        contrib_ratios, sampling_ratios = get_sampling_ratios(
            args.sample_labels, args.contrib_ratio, input_files=exp_files,
            total_counts=args.total_counts, pe_adjustment=args.paired_end_adjust, raw_format="tagAlign")
        downsampled_files = run_down_sampling(
            sample_labels=args.sample_labels, input_files=exp_files, sampling_ratios=sampling_ratios,
            chrom_size=args.chrom_size, random_generator=random_g, raw_format=args.raw_format, threads=args.threads
        )
        # simulate bulk tracks
        exp_bulk = simulate_bulk_tags(files=downsampled_files["ds-frags"].values(),
                                      prefix=f"{args.experiment_name}.exp")
        if args.raw_control_frags is not None and len(args.raw_control_frags) > 0:
            control_files = {
                s: (args.raw_control_frags[idx],) for idx, s in enumerate(args.sample_labels)}
            _, control_sampling_ratios = get_sampling_ratios(
                args.sample_labels, args.contrib_ratio, input_files=control_files,
                total_counts=args.total_control_counts, pe_adjustment=args.paired_end_adjust_control, raw_format="tagAlign")
            control_downsampled_files = run_down_sampling(
                sample_labels=args.sample_labels, input_files=control_files, sampling_ratios=control_sampling_ratios,
                chrom_size=args.chrom_size, random_generator=random_g, raw_format=f"control-{args.raw_format}",
                threads=args.threads, suffix=".ctl"
            )
            ct_bulk = simulate_bulk_tags(files=control_downsampled_files["ds-frags"].values(),
                                         prefix=f"{args.experiment_name}.ctl")
        else:
            logger.info("No control fragment was specified, skip downsampling for control")
            ct_bulk = None

        pl_bulk = simulate_bulk_tracks(
            files=downsampled_files["pl"].values(),
            chrom_sizes=args.chrom_size,
            prefix=f"{args.experiment_name}.pl", scale=1,
            to_int=True, wiggle_exe=args.wiggletools,
            wig2bw_exe=args.wig2bw
        )
        mn_bulk = simulate_bulk_tracks(
            files=downsampled_files["mn"].values(),
            chrom_sizes=args.chrom_size,
            prefix=f"{args.experiment_name}.mn", scale=1,
            to_int=True, wiggle_exe=args.wiggletools,
            wig2bw_exe=args.wig2bw
        )
        # call peaks
        if args.peaks is None:
            mean_fl = int(np.average(args.macs2_fraglen)) if args.macs2_fraglen else None
            peaks = call_macs2(
                ta=exp_bulk, ctl_ta=ct_bulk, gensz=args.macs2_gs, pval_thresh=args.macs2_pval,
                qval_thresh=args.fdr_target, frag_len=mean_fl,
                prefix=args.experiment_name, macs2_exe=args.macs2_exec)
        else:
            peaks = args.peaks
        try:
            os.remove(exp_bulk)
            os.remove(ct_bulk)
        except:
            pass
    else:
        raise ValueError(f"Unsupported raw format {args.raw_format}")

    # downsample ATAC
    if not args.skip_downsampling:
        rel_atac_contribs = {sl: sr / sum(args.atac_contrib_ratio) for sl, sr in
                             zip(args.sample_labels, args.atac_contrib_ratio)}
        sc_atac, sc_atac_meta = downsample_scatac_libs(
            total_cells=args.total_cells, cell_ratios=rel_atac_contribs,
            sample_labels=args.sample_labels, sc_atac=args.sc_atac,
            mean_frags_per_cell=args.mean_frags_per_cell, std_frags_per_cell=args.std_frags_per_cell,
            cell_barcodes=args.cell_barcodes, chrom_size=args.chrom_size, threads=args.threads,
            random_generator=random_g, frag_proc=args.frag_processing, rpm_norm=args.rpm_norm,
        )
        sc_atac_meta.to_csv("scatac.meta.csv", index=False, header=False)
        # RPM norm
        sc_atac_meta["rpm"] = (1000 * 1000) / sc_atac_meta[1]
        sc_atac_meta[[0, "rpm"]].to_csv("scatac.norm.csv", index=False, header=False)
        norm = sc_atac_meta[[0, "rpm"]]
    else:
        sc_atac = args.sc_atac
        # infer the type of sc_atac input
        if args.sc_atac[0].find(".tsv.gz") != -1:
            sc_atac = []
            n_frags_per_sample = np.zeros(n_samples, dtype=float)
            for i, scf in enumerate(args.sc_atac):
                _tmp_frag_df = pd.read_csv(scf, sep="\t")
                n_frags_per_sample[i] = _tmp_frag_df.shape[0]
                sc_atac.append(
                    frag_file_to_bw(fragment_file=scf, frag_proc=args.frag_processing, chrom_size=args.chrom_size,
                                    n_frags=_tmp_frag_df.shape[0], rpm_norm=args.rpm_norm, save_to_workdir=True))
            norm = pd.DataFrame({
                0: [os.path.split(f)[1] for f in args.sc_atac],
                "rpm": (1000.*1000.) / n_frags_per_sample
            })
        else:
            # if files are already in bigWig format... we, do nothing...
            sc_atac = args.sc_atac
            norm = None

    # build dataset
    if args.build_hdf5:
        prepare_dataset(**{
            "regions": peaks,
            "final_regions": args.final_peaks,
            "merge_overlap_peaks": args.merge_overlap_peaks,
            "background_sampling_ratio": args.background_sampling_ratio,
            "background_blacklist": args.background_blacklist,
            "bulk_pl": pl_bulk,
            "bulk_mn": mn_bulk,
            "target_sliding_sum": args.target_sliding_sum,
            "accessibility": sc_atac,
            "save_to": ".",
            "genome_fa": args.genome_fa,
            "chrom_size": args.chrom_size,
            "window_size": args.window_size,
            "seed": args.random_seed,
            "ref_labels": args.sample_labels,
            "ref_pls": [downsampled_files["pl"][s] for s in args.sample_labels],
            "ref_mns": [downsampled_files["mn"][s] for s in args.sample_labels if s in downsampled_files["mn"]],
        })

        # append scAcc norms and/or prior to the h5 dataset
        transformed_norm = None
        if norm is not None:
            transformed_norm = norm.copy()
            _mapping = {v: k for k, v in enumerate(norm[0].unique())}
            transformed_norm[0] = transformed_norm[0].map(_mapping)

        prior_mat = None
        if args.prior is not None and os.path.exists(args.prior):
            prior_df = pd.read_csv(args.prior, sep="\t", index_col=0)
            assert prior_df.shape[0] == prior_df.shape[1] == len(args.sample_labels)
            assert all([v == args.sample_labels[k] for k, v in enumerate(prior_df.columns)])
            assert all([v == args.sample_labels[k] for k, v in enumerate(prior_df.index)])
            prior_mat = prior_df.to_numpy()

        with (h5py.File("./data.h5", "a") as fh):
            # write cluster labels
            gr = fh["dec"]
            gr.attrs["cluster_names"] = ",".join(args.sample_labels)

            # save norm factors if any
            if transformed_norm is not None or prior_mat is not None:
                if transformed_norm is not None:
                    ds = fh.create_dataset("scatac_norm", data=transformed_norm.to_numpy())
                    for cluster_str, cluster_idx in _mapping.items():
                        ds.attrs[f"c_{cluster_idx}"] = cluster_str
                if prior_mat is not None:
                    ds = fh.create_dataset("dec/prior", data=prior_mat)
