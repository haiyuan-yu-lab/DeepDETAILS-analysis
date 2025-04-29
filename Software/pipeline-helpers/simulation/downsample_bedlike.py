#!/usr/bin/env python
import argparse
import os

import numpy as np
import pandas as pd
from typing import Union
from utils import bed_to_cov_bw


def downsample_bed(file_path: str, keep_prob: float, save_to: str,
                   scope: Union[None, set], random_generator: Union[None, np.random._generator.Generator],
                   is_paired_tagalign: bool = False) -> str:
    """
    Downsample reads saved in a bed file.

    Parameters
    ----------
    file_path : str
        Path to the input bed file
    keep_prob : float
        Probability to keep a read (the prob of success in binomial dist)
        If the value is not smaller than 1, no sampling will be performed.
    save_to : str
        Path to write the down-sampled bed file
    scope : Union[None, set]
        Chromosomes to be included. If set as None, all chromosomes will be included.
    random_generator : Union[None, np.random._generator.Generator]
        Random number generator
    is_paired_tagalign : bool
        If the input file is a paired tagAlign file, which stores
        read 1 and read 2 in two adjacent lines.

    Notes
    -----
    This implementation consumes significant amount of memory as it reads in the entire bed file.

    Returns
    -------
    sampled_file : str
        Path to the sampled file
    """
    rng = np.random if random_generator is None else random_generator
    df = pd.read_csv(file_path, sep="\t", header=None, comment="#")
    if df.shape[1] < 3:
        raise ValueError("Expect the input bed file to have at least 3 columns")
    if scope is not None:
        df = df.loc[df[0].isin(scope)]
    if keep_prob < 1:
        if is_paired_tagalign:
            if df.shape[0] % 2:
                raise ValueError("Input file is told to be in paired-end tagAlign, "
                                 "which should have even number of lines. Odd lines observed.")
            n_pairs = df.shape[0] // 2
            paired_index = rng.shuffle(np.arange(n_pairs))[:int(n_pairs * keep_prob)]
            selection_index = []
            for pi in paired_index:
                selection_index.append(pi)
                selection_index.append(pi+1)
            df = df.loc[selection_index]
        else:
            df = df.loc[rng.random(df.shape[0]) < keep_prob]
    else:
        print("Oversampling from the input bed file since the keep_prob is greater than 1.")
        if is_paired_tagalign:
            if df.shape[0] % 2:
                raise ValueError("Input file is told to be in paired-end tagAlign, "
                                 "which should have even number of lines. Odd lines observed.")
            n_pairs = df.shape[0] // 2
            n_targets = int(n_pairs * keep_prob)
            paired_index = rng.choice(np.arange(n_pairs), size=n_targets, replace=True)
            selection_index = []
            for pi in paired_index:
                selection_index.append(pi)
                selection_index.append(pi+1)
            df = df.loc[selection_index]
        else:
            n_targets = int(df.shape[0] * keep_prob)
            df = df.sample(n=n_targets, replace=True, random_state=rng).reset_index(drop=True)
    df.sort_values([0, 1], inplace=True)
    df.to_csv(save_to, sep="\t", header=False, index=False)
    return save_to


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inputs", nargs="+", required=True,
                        help="Input tagAlign/Bed files")
    parser.add_argument("-s", "--scale-factors", nargs="+", type=float, required=True,
                        help="Scale factors for each file defined in --inputs")
    parser.add_argument("-l", "--labels", nargs="+", type=str, required=True,
                        help="Labels for each file defined in --inputs")
    parser.add_argument("-x", "--comments", default="#",
                        help="Lines starting with this will be ignored. "
                             "Useful if you are working with fragments files yield by 10x pipelines.")
    parser.add_argument("-r", "--random-seed", type=int,
                        help="Seed for random sampling")
    parser.add_argument("-o", "--save-to", type=str,
                        help="All output files will be written into this place")
    parser.add_argument("-c", "--chrom-size", type=str,
                        help="Path to a tab-separated file which defines the sizes of chromosomes")
    parser.add_argument("-k", "--keep-bed", action="store_true", default=False,
                        help="Set this trigger if you want to keep the sampled bed files.")

    args = parser.parse_args()

    assert len(args.inputs) == len(args.scale_factors) == len(args.labels)

    random_g = np.random.default_rng(args.random_seed)
    chr_pool = set(pd.read_csv(args.chrom_size, sep="\t", header=None)[0].values)
    for i, in_bedfile in enumerate(args.inputs):
        print(in_bedfile)
        output_file_prefix = os.path.join(
            args.save_to,
            args.labels[i]  # (root, ext)
        )
        # first downsample the bed file
        sampled_bed = output_file_prefix + ".sampled.bed"
        output_bw_file = output_file_prefix + ".bw"
        if not os.path.exists(output_bw_file):
            downsample_bed(
                in_bedfile,
                keep_prob=args.scale_factors[i],
                save_to=sampled_bed,
                random_generator=random_g,
                scope=chr_pool
            )
            # convert the sampled bed to coverage bigWig

            bed_to_cov_bw(sampled_bed, output_bw_file, args.chrom_size)

            # clean up
            if not args.keep_bed:
                os.remove(sampled_bed)
        else:
            print(f"File {output_bw_file} exists, skipping...")
