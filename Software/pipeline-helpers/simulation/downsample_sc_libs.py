import argparse
import logging
import os
import pybedtools
import numpy as np
import pandas as pd
from typing import Union
from multiprocessing import Pool
from utils import bed_to_cov_bw

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO)


def frag_to_cut_sites(frag_df: Union[pd.DataFrame, str], save_to: str, chrom_size: str, window_size: int = 150,
                      dual_directions: bool = False):
    """Convert fragments to cut sizes

    Parameters
    ----------
    frag_df : Union[pd.DataFrame, str]

    save_to : str

    chrom_size: str

    window_size : int
        Window size for cute sites. Set as 150 to mimic the ENCODE ATAC-seq pipeline.
        Set as 400 to mimic the CellRanger ATAC (CRA) pipeline
    dual_directions : bool
        If True, the function will extract cut sites from
        both the start and end sides of the fragments, this mimics
        CRA's behavior.

    Returns
    -------

    """
    current_tmp_dir = pybedtools.get_tempdir()
    # if pybedtools is using the system's default tmp dir.,
    # then switch to current working directory to avoid using up all spaces at `/`
    if current_tmp_dir in ("/tmp", "/var/tmp", "/usr/tmp", "C:\\TEMP", "C:\\TMP", "\\TMP"):
        pybedtools.set_tempdir(".")
    half_window = window_size // 2
    if isinstance(frag_df, pd.DataFrame):
        frag_bed = pybedtools.BedTool.from_dataframe(frag_df)
    else:
        frag_bed = pybedtools.BedTool(frag_df)
    start_side = frag_bed.flank(l=half_window, r=0, g=chrom_size).slop(r=half_window, l=0, g=chrom_size)
    if dual_directions:
        end_side = frag_bed.flank(r=half_window, l=0, g=chrom_size).slop(l=half_window, r=0, g=chrom_size)
        pybedtools.BedTool.cat(
            *[start_side, end_side], postmerge=False
        ).sort().saveas(save_to)
    else:
        start_side.sort().saveas(save_to)


def frag_file_to_bw(fragment_file: str, frag_proc: str, chrom_size: str, n_frags: int,
                    rpm_norm: bool = False, save_to_workdir=False) -> str:
    """
    
    Parameters
    ----------
    fragment_file : str
        A compressed tsv file storing fragments.
    frag_proc : str
        Additional processes to the fragment file. Allowed types:
        * naive: no additional processing
        * encode: extend 150 bp from the cut site
        * cellranger: extend 400 bp from the cut site
        * 5pi: only the 5' insert site
    chrom_size : str
        A tab-separated file with chromosome sizes.
    n_frags : int
        Number of fragments in the fragment file. Affective only when rpm_norm is True.
    rpm_norm : bool, optional
        Whether to store RPM-normalized signal in the bigWig output
    save_to_workdir : bool, optional
        Force saving the bigWig output to the working directory; otherwise, it will be written to the
        folder containing the input fragment_file.

    Returns
    -------
    save_to : str
        Saved bigWig file
    """
    report_5p_cov = False
    if frag_proc == "naive":
        csb_file = fragment_file
    elif frag_proc == "encode":
        csb_file = fragment_file.replace("tsv.gz", "cs.bed")
        frag_to_cut_sites(fragment_file, csb_file, chrom_size, 150, False)
    elif frag_proc == "cellranger":
        csb_file = fragment_file.replace("tsv.gz", "cs.bed")
        frag_to_cut_sites(fragment_file, csb_file, chrom_size, 400, True)
    elif frag_proc == "5pi":  # 5' insert sites
        csb_file = fragment_file
        report_5p_cov = True
    else:
        logger.exception(f"Unsupported {frag_proc}")
        csb_file = fragment_file

    if save_to_workdir:
        _, tmp = os.path.split(fragment_file)
    else:
        tmp = fragment_file
    save_to = tmp.replace("tsv.gz", "bw")
    if chrom_size is not None:
        bed_to_cov_bw(
            csb_file, save_to, chrom_size,
            rpm_norm=n_frags if rpm_norm else None,
            report_5p_cov=report_5p_cov)
    return save_to


def downsample_sclib(fragment_file: str, n_cells: int, mean_frags_per_cell: int, std_frags_per_cell: float,
                     save_to: str, allowed_barcodes: Union[list, str, None], chrom_size: Union[str, None],
                     random_seed: Union[int, None, callable], frag_proc: str = "naive",
                     rpm_norm: bool = False) -> int:
    """Downsample a single-cell sequencing library

    It first samples cells, then samples fragments from these selected cells.

    Parameters
    ----------
    fragment_file : str
        Fragment file (bed-like)
    n_cells : int
        Number of cells to be sampled
    mean_frags_per_cell : int
        This number defines the average of fragments one cell will have in the downsampled library
    std_frags_per_cell : float
        Std of the frags per cell in the downsampled library. For example, if you want cells in the downsampled
        lib to have mean +/- 500, according to the 3-sigma limits, you can set this as 500/3
    save_to : str
        File name of the sampled library
    allowed_barcodes : Union[list, str, None]
        Allowed cell barcodes. Can be a list of barcodes, a str to a file containing the list of allowed cell barcodes.
        If None, all barcodes can be used.
    chrom_size : Union[str, None]
        If a path to the chromosome size file is provided,
    random_seed : Union[int, None, callable]
        Random seed for sampling.
    frag_proc : str
        Fragment processing strategy. Supported values: naive, encode, cellranger.
    rpm_norm : bool
        Export RPM-normalized signal tracks

    Raises
    ------
    ValueError
        If the input fragment file doesn't have exactly 5 columns.

    Returns
    -------
    n_frags : int
        Number of sampled fragments
    """
    if random_seed is None:
        random_g = np.random
    elif isinstance(random_seed, int):
        random_g = np.random.default_rng(random_seed)
    else:
        random_g = random_seed
    logger.info(f"{save_to}: Downsampling fragments file {fragment_file} to "
                f"{n_cells} cells with mean frags per cell {mean_frags_per_cell}, "
                f"and std frags per cell {std_frags_per_cell}")
    sampling_targets = np.round(random_g.normal(mean_frags_per_cell, std_frags_per_cell, n_cells)).astype(int)

    # load frags
    logger.info(f"{save_to}: Loading fragments")
    frags = pd.read_csv(fragment_file, sep="\t", header=None, comment="#")
    if frags.shape[1] != 5:
        raise ValueError("Expecting the fragments file to have 5 columns")
    logger.info(f"{save_to}: Loaded {frags.shape[0]} fragments")

    # if allowed_barcodes is a file like the one from 10x's filtered_peak_bc_matrix/barcodes.tsv
    if isinstance(allowed_barcodes, str):
        # filtered_peak_bc_matrix/barcodes.tsv
        _df = pd.read_csv(allowed_barcodes, sep="\t", header=None)
        allowed_barcodes = _df[0].values
    elif allowed_barcodes is None:
        logger.warning(f"Library {fragment_file} will not be downsampled using cell barcodes.")
        allowed_barcodes = frags[3].values

    # get fragments in each cell
    frags_per_cell = frags[[3]].groupby(3).groups

    # sampling cells
    usable_cells = len(allowed_barcodes)
    logger.info(f"{save_to}: {usable_cells} allowed cell barcodes")
    if n_cells > usable_cells:
        logger.debug(
            f"{save_to}: Trying to sample {n_cells} cells, "
            f"which is larger than the number of usable cells ({usable_cells}) in the lib")
        n_cells = usable_cells
    selected_cell_barcodes = random_g.choice(allowed_barcodes, n_cells, replace=False)

    # sampling fragments
    sampled_frag_indexes = []
    for i, scb in enumerate(selected_cell_barcodes):
        cell_frags = len(frags_per_cell[scb])
        if cell_frags < sampling_targets[i]:
            logger.debug(f"{save_to}: Cell {scb} only has {cell_frags} fragments, "
                         "adjusting sampling target for this cell...")
        st = min(cell_frags, sampling_targets[i])
        sampled_frag_indexes.extend(random_g.choice(frags_per_cell[scb], st, replace=False))

    # write the sampled fragments to a file
    logger.info(f"{save_to}: Sorting and exporting")
    frags.loc[sampled_frag_indexes].sort_values([0, 1]).to_csv(save_to, sep="\t", header=False, index=False)

    logger.info(f"{save_to}: Done sampling ({len(sampled_frag_indexes)} fragments)")
    n_frags = len(sampled_frag_indexes)
    frag_file_to_bw(save_to, frag_proc, chrom_size, n_frags, rpm_norm, save_to_workdir=False)
    logger.info(f"{save_to}: Done track building")
    return n_frags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--fragments", nargs="+", required=True,
                        help="Fragments files, multiple files should be separated by spaces")
    parser.add_argument("-o", "--output-prefixes", nargs="+", required=True,
                        help="Output prefixes for each library. The number should match the number of fragments libs")
    parser.add_argument("-c", "--cell-barcodes", nargs="*",
                        help="Allowed cell barcodes for each library (barcodes that passing filters)."
                             "All barcodes will be allowed if nothing is specified.")
    parser.add_argument("-n", "--n-cells-per-sample", nargs="+", type=int, required=True,
                        help="Number of cells to be sampled from each sample. Should be either a single value or"
                             "a list of values for each library")
    parser.add_argument("-m", "--mean-frags-per-cell", nargs="+", type=int, required=True,
                        help="Average number of fragments to be sampled from each cell. "
                             "Should be either a single value or a list of values for each library")
    parser.add_argument("-s", "--std-frags-per-cell", nargs="+", type=float, required=True,
                        help="Standard deviation of fragments to be sampled from each cell."
                             "Should be either a single value or a list of values for each library")
    parser.add_argument("-r", "--random-seed", type=int, default=1, required=False,
                        help="Random seed for sampling")
    parser.add_argument("-p", "--repeat-times", type=int, default=1, required=False,
                        help="How many samples you want to draw?")
    parser.add_argument("-t", "--tasks", type=int, default=4, required=False,
                        help="Number of parallel tasks allowed")
    parser.add_argument("-x", "--chrom-size", type=str, required=False,
                        help="Path to a file defining the sizes of all chromosomes."
                             "If provided, this tool will convert all downsampled libraries to coverage tracks.")
    parser.add_argument("--frag-processing", type=str, required=False,
                        choices=("naive", "encode", "cellranger"), )
    parser.add_argument("--rpm-norm", action="store_true", )

    args = parser.parse_args()

    if len(args.fragments) != len(args.output_prefixes):
        raise argparse.ArgumentTypeError("-f, and -o should have identical lengths")
    if len({len(args.mean_frags_per_cell), len(args.std_frags_per_cell), len(args.n_cells_per_sample)}) != 1:
        raise argparse.ArgumentTypeError("-n, -m, and -s should have identical lengths")
    if len(args.mean_frags_per_cell) == 1:  # broadcasting
        args.mean_frags_per_cell = args.mean_frags_per_cell * len(args.fragments)
        args.std_frags_per_cell = args.std_frags_per_cell * len(args.fragments)
        args.n_cells_per_sample = args.n_cells_per_sample * len(args.fragments)
    if args.cell_barcodes is None:
        args.cell_barcodes = [None, ] * len(args.fragments)
    if len(args.cell_barcodes) != len(args.fragments):
        raise argparse.ArgumentTypeError("-c should have either the same number of values as -f or zero value")

    jobs = []
    for r in range(args.repeat_times):
        for i, frag_file in enumerate(args.fragments):
            jobs.append((
                frag_file, args.n_cells_per_sample[i], args.mean_frags_per_cell[i],
                args.std_frags_per_cell[i], f"{args.output_prefixes[i]}.s{r}.tsv.gz",
                args.cell_barcodes[i], args.chrom_size, args.random_seed + r,
                args.frag_processing, args.rpm_norm))

    with Pool(args.tasks) as p:
        p.starmap(downsample_sclib, jobs)
