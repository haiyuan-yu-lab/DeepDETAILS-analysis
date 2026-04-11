import argparse
import logging
import os
import re
import pybedtools
import pyBigWig
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Union, Optional, Sequence
from torch.utils.data import Dataset, DataLoader


logging.basicConfig(format="%(name)s - %(asctime)s - %(levelname)s: %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S",
                    level=logging.INFO,
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger("DeepDETAILS - Supporting analysis - Dominant Position Correction")


class MockBW:
    @staticmethod
    def values(chrom: Any, start: Any, end: Any):
        return np.zeros(end - start)

    def close(self):
        pass


def open_bw_safe(in_file):
    try:
        bw = pyBigWig.open(in_file)
    except RuntimeError as e:
        logger.warning(f"Cannot open {in_file} using mock instead: {e}")
        bw = MockBW()
    return bw


class EvaluationDataset(Dataset):
    def __init__(self, pl_pred_bw_file: str, pl_obs_bw_file: str, peak_file: Union[str, pd.DataFrame],
                 mn_pred_bw_file: Optional[str] = None, mn_obs_bw_file: Optional[str] = None,
                 unified_len_def: int = 1000, distal_only: int = 0, all_regions: bool = False,
                 promoter_file: Optional[str] = None):
        """

        Parameters
        ----------
        pl_pred_bw_file : str
            Path to the bigwig storing predictions for the forward strand
        pl_obs_bw_file : str
            Path to the bigwig storing ground-truth for the forward strand
        peak_file : str
            Path to a headless-csv file storing the regions of interest
        mn_pred_bw_file : str
            Path to the bigwig storing predictions for the reverse strand
        mn_obs_bw_file : str
            Path to the bigwig storing ground-truth for the reverse strand
        unified_len_def : int
            Extend/shrink the spans of the regions to this length
        distal_only : int
            * 1: use only distal regions
            * 0: use all regions
            * -1 : use only proximal regions
        all_regions : bool
            Use all regions (peak and background regions)
        promoter_file : Optional[str]


        """
        if any([not os.path.exists(f) for f in (
                pl_pred_bw_file, pl_obs_bw_file)]):
            raise IOError()
        self.pl_pred_bw = open_bw_safe(pl_pred_bw_file)
        self.pl_obs_bw = open_bw_safe(pl_obs_bw_file)
        self._in_files = [pl_pred_bw_file, pl_obs_bw_file, mn_pred_bw_file, mn_obs_bw_file]

        if all([f is not None and os.path.exists(f) for f in (
                mn_pred_bw_file, mn_obs_bw_file)]):
            self.mn_pred_bw = open_bw_safe(mn_pred_bw_file)
            self.mn_obs_bw = open_bw_safe(mn_obs_bw_file)
        else:
            self.mn_pred_bw = None
            self.mn_obs_bw = None
            print("No signal tracks for the reverse strands.")

        if isinstance(peak_file, str):
            self.regions = pd.read_csv(peak_file, header=None, comment="#")
        else:
            self.regions = peak_file.copy()
        if not all_regions:
            self.regions = self.regions.loc[self.regions[3] == 1].reset_index(drop=True)
        allowed_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", ]
        self.regions = self.regions.loc[self.regions[0].isin(allowed_chroms)].reset_index(drop=True)

        if distal_only != 0 and promoter_file is not None:
            promoters = pybedtools.BedTool(promoter_file)
            if distal_only == 1:  # distal only
                self.regions = pybedtools.BedTool.from_dataframe(
                    self.regions).intersect(promoters, v=True).to_dataframe(
                    disable_auto_names=True, header=None)
            elif distal_only == -1:  # proximal only
                self.regions = pybedtools.BedTool.from_dataframe(
                    self.regions).intersect(promoters, u=True).to_dataframe(
                    disable_auto_names=True, header=None)
        half = unified_len_def // 2
        mids = (self.regions[1] + self.regions[2]) // 2
        self.regions[1] = mids - half
        self.regions[2] = mids + half

        self._warns = 0

    def __len__(self):
        return self.regions.shape[0]

    def bw_retriever(self, bw_in, chrom, start, end, bw_idx=None):
        try:
            return np.abs(np.nan_to_num(bw_in.values(chrom, start, end)))
        except Exception:
            if bw_idx is not None:
                bw_info = self._in_files[bw_idx]
            else:
                bw_info = str(bw_in)
            self._warns += 1
            if self._warns < 3:
                print(f"Warning: No values stored in {bw_info} for region {chrom}:{start}-{end}")
            return np.zeros(end - start)

    def __getitem__(self, idx: int):
        try:
            y_hats = []
            y = []
            have_mn = False
            hit = self.regions.iloc[idx]
            y_hat_pl = self.bw_retriever(self.pl_pred_bw, hit[0], hit[1], hit[2], 0)
            y_pl = self.bw_retriever(self.pl_obs_bw, hit[0], hit[1], hit[2], 1)

            if self.mn_pred_bw is not None and self.mn_obs_bw is not None:
                y_hat_mn = self.bw_retriever(self.mn_pred_bw, hit[0], hit[1], hit[2], 2)
                y_mn = self.bw_retriever(self.mn_obs_bw, hit[0], hit[1], hit[2], 3)
                have_mn = True

            y_hats.append(y_hat_pl)
            y.append(y_pl)

            if have_mn:
                y_hats.append(y_hat_mn)
                y.append(y_mn)
        except Exception as e:
            logger.warning(hit)
            raise e

        return torch.from_numpy(np.stack(y_hats)), torch.from_numpy(np.stack(y))

    def close_files(self):
        self.pl_pred_bw.close()
        self.pl_obs_bw.close()
        if self.mn_pred_bw is not None:
            self.mn_pred_bw.close()
        if self.mn_obs_bw is not None:
            self.mn_obs_bw.close()


def get_dominant_positions(
        y: torch.Tensor, y_hat: torch.Tensor, summit_threshold:Optional[int] = 10
) -> tuple[list[int], list[int]]:
    """
    Get dominant positions for a set of regions (m).
    If there's no signal in `y` that's larger than `summit_threshold`,
    this function consider that region as background regions,
    and the region will be discarded in the analysis

    Parameters
    ----------
    y : torch.Tensor
        Ground truth (m, s, l)
    y_hat : torch.Tensor
        Predictions (m, s, l)
    summit_threshold : Optional[int]
        Minimum signal to consider region as meaningful

    Notes
    -----
    This function assumes input values are non-negative

    Returns
    -------
    obs_summits : list[int]
        Observed summit positions (<= s*m)
    pred_summits : list[int]
        Predicted summit positions
    """
    # find arg on each sample, each strand, then flatten
    y_loc = y.argmax(axis=-1).flatten()
    y_max = y.max(dim=-1)[0].flatten()
    probe = y_max > summit_threshold
    # discard summits where the signal intensity is low
    y_loc = y_loc[probe]
    obs_summits = y_loc.tolist()
    y_hat_loc = y_hat.argmax(axis=-1).flatten()
    y_hat_loc = y_hat_loc[probe]
    pred_summits = y_hat_loc.tolist()

    return obs_summits, pred_summits


def evaluate_core(
    pred_pl_file: str, pred_mn_file: str, obs_pl_file: str, obs_mn_file: str, peak_file: str,
    use_all_regions: bool, summit_threshold: Optional[int] = 10, batch_size: Optional[int] = 32,
    promoter_file: Optional[str] = None
):
    """Core evaluation function

    Parameters
    ----------
    pred_pl_file : str
        Path to the prediction file (plus strand)
    pred_mn_file : str
        Path to the prediction file (minus strand)
    obs_pl_file : str
        Path to the real observation file (plus strand)
    obs_mn_file : str
        Path to the real observation file (minus strand)
    peak_file : str
        Path to the peak file (bed format)
    use_all_regions : bool
        Whether to use all regions
    summit_threshold : Optional[int]
        Minimum signal to consider region as meaningful
    batch_size : Optional[int]
        Batch size
    promoter_file : Optional[str]


    Returns
    -------

    """
    assert os.path.isfile(pred_pl_file), pred_pl_file
    assert os.path.isfile(pred_mn_file), pred_mn_file
    assert os.path.isfile(obs_pl_file), obs_pl_file
    assert os.path.isfile(obs_mn_file), obs_mn_file
    assert os.path.isfile(peak_file), peak_file
    if promoter_file is not None:
        assert os.path.isfile(promoter_file), promoter_file

    ds = EvaluationDataset(
        pl_pred_bw_file=pred_pl_file, mn_pred_bw_file=pred_mn_file,
        pl_obs_bw_file=obs_pl_file, mn_obs_bw_file=obs_mn_file,
        peak_file=peak_file, all_regions=use_all_regions,
        promoter_file=promoter_file)

    obs_summits = []
    pred_summits = []
    _iter = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    for batch in _iter:
        y_hat = batch[0]
        y = batch[1]
        obss, prds = get_dominant_positions(y, y_hat, summit_threshold=summit_threshold)
        obs_summits.extend(obss)
        pred_summits.extend(prds)
    ds.close_files()

    dom_pos_df =  pd.DataFrame({
        0: np.asarray(obs_summits),
        1: np.asarray(pred_summits)
    })
    return dom_pos_df, np.corrcoef(dom_pos_df.T)[0, 1]


def main(
    pred_pl_files: Sequence[str], pred_mn_files: Sequence[str],
    obs_pl_files: Sequence[str], obs_mn_files: Sequence[str],
    peak_file: str, prefix: str, plot_heatmap: Optional[bool] = True,
    use_all_regions: Optional[bool] = False, n_samples: Optional[int] = 5000,
    random_seed: Optional[int] = 12345, summit_threshold: Optional[int] = 10,
    batch_size: Optional[int] = 32, promoter_file: Optional[str] = None,
    tasks: Optional[int] = 16):
    """
    main function

    Parameters
    ----------
    pred_pl_files : Sequence[str]
        Paths to the prediction files (plus strand)
    pred_mn_files : Sequence[str]
        Paths to the prediction files (minus strand)
    obs_pl_files : Sequence[str]
        Paths to the real observation files (plus strand)
    obs_mn_files : Sequence[str]
        Paths to the real observation files (minus strand)
    peak_file : str
        Path to the peak file (bed format)
    prefix : str
        Output prefix (full path + prefix)
    plot_heatmap : Optional[bool]
        Plot heatmap
    use_all_regions : Optional[bool]
        Use all regions
    n_samples : Optional[int]
        Number of samples for plotting and calculating the
        correlation coefficient in the final merged sample
    random_seed : Optional[int]
        Random seed for sampling
    summit_threshold : Optional[int]
        Minimum signal to consider region as meaningful
    batch_size : Optional[int]
        Batch size
    promoter_file : Optional[str]
        Path to a promoter file (bed format)
    tasks : Optional[int]
        Number of tasks for multi-processing

    Returns
    -------

    """
    jobs = []

    for ppf, pmf, opf, omf in zip(pred_pl_files, pred_mn_files, obs_pl_files, obs_mn_files, strict=True):
        jobs.append((ppf, pmf, opf, omf, peak_file, use_all_regions, summit_threshold, batch_size, promoter_file))

    with Pool(tasks) as pool:
        pool_results = pool.starmap(evaluate_core, jobs)

    result_rows = []
    per_rep_results = defaultdict(list)
    regex = r"log_(r\d+)"
    for job_confs, job_results in zip(jobs, pool_results, strict=True):
        logger.info(f"{job_confs[0]}\t{job_confs[1]}\t{job_confs[2]}\t{job_confs[3]}\t{job_results[1]}")
        items = list(job_confs[:4])
        items.append(job_results[1])
        result_rows.append(items)

        hits = re.findall(regex, job_confs[0])
        if len(hits) > 0:
            logger.info(f"prediction pair {job_confs[0]} and {job_confs[1]} will be categorized into {hits[0]}")
            per_rep_results[hits[0]].append(job_results[0])

    merged_data = []
    if len(per_rep_results) > 0:
        for rep_name, rep_results in per_rep_results.items():
            merged_dom_pos = pd.concat(rep_results, ignore_index=True)
            merged_data.append(merged_dom_pos)
            merged_dom_pos.to_csv(f"{prefix}.dom_pos.{rep_name}.csv.gz", compression="gzip", index=False)
    else:
        merged_dom_pos = pd.concat([pr[0] for pr in pool_results], ignore_index=True)
        merged_data.append(merged_dom_pos)
        merged_dom_pos.to_csv(f"{prefix}.merged_dom_pos.csv.gz", compression="gzip", index=False)


    for idx, merged_df in enumerate(merged_data):
        plot_df = merged_df.sample(n_samples, random_state=random_seed)
        corr = np.corrcoef(plot_df[[0, 1]].T)[0, 1]
        result_rows.append(("Per run", idx, idx, idx, corr))
        logger.info(f"Correlation ({idx}): {corr:.4f}")

        if plot_heatmap:
            fig = plt.figure(figsize=(1.7, 1.2))
            ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
            log_density = ax.scatter_density(
                x=plot_df[0],
                y=plot_df[1],
            )

            ax.text(
                0.05, 0.95,
                fr'$r=${corr:.3f}',
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left', fontsize=6,
            )

            fig.colorbar(log_density)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("Predicted")
            ax.set_xlabel("Observed")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.savefig(f"{prefix}.{idx}.pdf", bbox_inches="tight")
            plt.close(fig)

    pd.DataFrame(result_rows).to_csv(f"{prefix}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pred-pl-files", type=str, nargs="+", required=True)
    parser.add_argument("-m", "--pred-mn-files", type=str, nargs="+", required=True)
    parser.add_argument("-P", "--obs-pl-files", type=str, nargs="+", required=True)
    parser.add_argument("-M", "--obs-mn-files", type=str, nargs="+", required=True)

    parser.add_argument("-r", "--peak-file", required=True)
    parser.add_argument("-o", "--prefix", type=str, default="dpc")
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-s", "--summit-threshold", type=int, default=10)
    parser.add_argument("-R", "--promoter-file", type=str)
    parser.add_argument("-a", "--use-all-regions", action="store_true")
    parser.add_argument("-t", "--tasks", default=16)

    parser.add_argument("-n", "--no-heatmap", dest="plot_heatmap", action="store_false")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=12345)

    args = parser.parse_args()

    handler = logging.FileHandler(f"{args.prefix}.log")
    formatter = logging.Formatter("%(name)s - %(asctime)s - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    main(**vars(args))
