import argparse
import logging
import pyBigWig
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import cycle
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO)

# Functions make_track_values_dict and write_scores_to_bigwigs are modified from
# https://github.com/kundajelab/nascent_RNA_models/blob/main/src/utils/write_bigwigs.py


def make_track_values_dict(scores: np.ndarray, regions: pd.DataFrame, chrom: str, verbose: bool = False) -> dict:
    """
    Prepare bigWig track for a chromosome

    Parameters
    ----------
    scores : np.ndarray
        Score values for all chromosomes. Shape: (total_regions, seq_len) or (total_regions, )
    regions : pd.DataFrame
        DataFrame with columns: 0 (chromosome names), 1 (starts), 2 (ends).
        A view of the parent dataframe which only contains regions on the specified chromosome.
        Indexes are still their old indexes as in the parent dataframe (since the df is just a view),
        and the index values correspond to the rows in all_values.
    chrom : str
        Name of a chromosome to be used for building the track.
        Value should be in coords[0]
    verbose : bool, optional
        Set verbose to True to see a progress bar.

    Returns
    -------
    track_values : dict
        Dictionary containing position as key and average value as value

    References
    ----------

    """
    chroms = regions[0].unique()
    assert len(chroms) == 1 and chrom in chroms, f"regions dataframe should only contain records for chromosomes {chrom}"

    # Use defaultdict to simplify appending values to positions
    track_values = defaultdict(list)

    # Iterate through DataFrame rows using tqdm for progress visualization
    for i, (_, start, end) in tqdm(
            regions.iterrows(), total=regions.shape[0], desc=chrom, disable=not verbose):
        # scores for load-like objects will only be a scalar
        # put the scores in cycle, so we don't need to repeat the scalar
        values = scores[i] if scores.ndim == 2 else cycle([scores[i],])

        positions = np.arange(start, end)

        # Update defaultdict with position and corresponding value
        track_values.update((pos, track_values[pos] + [val]) for pos, val in zip(positions, values))

    # take the mean at each position, so that if there was overlap, the average value is used
    track_values = {key: np.mean(vals) if len(vals) > 1 else vals[0] for key, vals in track_values.items()}
    return track_values


def write_scores_to_bigwigs(scores: np.ndarray, peaks_file: str, save_to: str,
                            chrom_sizes_file: str, verbose: bool = False):
    """
    Write attribution scores to a bigWig file.

    Parameters
    ----------
    scores : np.ndarray
        Score values for all chromosomes. Shape: total_regions, seq_len
    peaks_file : str
        Path to a bed file containing the corresponding regions for each row in the scores array.
    save_to : str
        Full path (including file name) to save the bigWig file.
    chrom_sizes_file : str
        Path to a tab-delimited file describing the size (col 2) of each chromosome (col 1).
    verbose : bool, optional
        Set verbose to True to see a progress bar.

    Returns
    -------
    None
    """
    scores = scores.astype("float64")
    peaks = pd.read_csv(peaks_file, sep="\t", header=None)

    assert peaks.shape[0] == scores.shape[0]

    chrom_sizes = pd.read_csv(chrom_sizes_file, sep="\t", header=None)

    with pyBigWig.open(save_to, "w") as bw:
        bw.addHeader([tuple(p) for p in chrom_sizes.values.tolist()])

        for chrom, sub_df in peaks.groupby(0):
            logger.info(f"Adding signals on chromosome {chrom}...")

            track_values_dict = make_track_values_dict(scores, sub_df[[0, 1, 2]], chrom, verbose=verbose)
            num_entries = len(track_values_dict)

            starts = sorted(list(track_values_dict.keys()))
            ends = [position + 1 for position in starts]
            scores_to_write = [track_values_dict[key] for key in starts]

            assert len(scores_to_write) == len(starts) and len(scores_to_write) == len(ends) > 0

            bw.addEntries([chrom for _ in range(num_entries)],
                          starts, ends=ends, values=scores_to_write)
            logger.info(f"Signals on chromosome {chrom} added...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write attribution scores to a bigWig file.")
    parser.add_argument("-i", "--input",
                        help="Path to a file containing attribution scores (numpy array).")
    parser.add_argument("-c", "--chrom-sizes",
                        help="Path to a tab-delimited file describing the size of each chromosome.")
    parser.add_argument("-p", "--peaks",
                        help="Path to a bed file containing the corresponding regions for each row in the scores array.")
    parser.add_argument("-o", "--save-to",
                        help="Full prefix (including directory path) for the output bigWig file(s).")
    parser.add_argument("-t", "--score-type", choices=["seq", "atac", "load"], default="seq",
                        help="Type of the input scores. (seq, atac, load)")
    parser.add_argument("-v", "--verbose",
                        action="store_true", help="Set to enable verbose mode with progress bars.")
    args = parser.parse_args()

    raw_scores = np.load(args.input)["arr_0"]
    if args.score_type == "seq":
        # shape of sequence attributions scores: regions, 4, seq_len
        write_scores_to_bigwigs(
            raw_scores.sum(axis=1), peaks_file=args.peaks,
            save_to=f"{args.save_to}.bw",
            chrom_sizes_file=args.chrom_sizes, verbose=args.verbose)
    elif args.score_type == "atac":
        # shape of atac attributions scores: regions, clusters, seq_len
        for cluster_id in range(raw_scores.shape[1]):
            write_scores_to_bigwigs(
                raw_scores[:, cluster_id, :], peaks_file=args.peaks,
                save_to=f"{args.save_to}.C{cluster_id}.bw",
                chrom_sizes_file=args.chrom_sizes, verbose=args.verbose)
    elif args.score_type == "load":
        # shape of load attributions scores: regions, clusters
        for cluster_id in range(raw_scores.shape[1]):
            write_scores_to_bigwigs(
                raw_scores[:, cluster_id], peaks_file=args.peaks,
                save_to=f"{args.save_to}.C{cluster_id}.bw",
                chrom_sizes_file=args.chrom_sizes, verbose=args.verbose)
    else:
        raise ValueError(f"{args.score_type} not supported.")
