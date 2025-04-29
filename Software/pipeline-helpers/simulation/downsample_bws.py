#!/usr/bin/env python
import argparse
import os

import numpy as np
import pandas as pd
from typing import Union
from utils import bigwig_to_bedgraph, bedgraph_to_bigwig


def downsample_bedgraph(file_path: str, keep_prob: float, save_to: str,
                        random_generator: Union[None, np.random._generator.Generator], use_poisson: bool = False):
    """
    Downsample signal values defined in a bedGraph file

    Parameters
    ----------
    file_path : str
        Path to the input bedGraph file
    keep_prob : float
        Probability to keep a read (the prob of success in binomial dist)
    save_to : str
        Path to write the down-sampled bedGraph file
    random_generator : Union[None, np.random._generator.Generator]
        Random number generator
    use_poisson : bool
        Use a Poisson sampler instead of the default Binomial sampler

    Returns
    -------

    """
    rng = np.random if random_generator is None else random_generator
    df = pd.read_csv(file_path, sep="\t", header=None)
    if df.shape[1] != 4:
        raise ValueError("Expect the input bedgraph file to have exactly 4 columns")

    # for negative counts, treat them as positive values first
    # then flip the signs back
    if use_poisson:
        downsampled_counts = np.sign(df[3]) * rng.poisson(df[3].abs() * keep_prob)
    else:
        downsampled_counts = np.sign(df[3]) * rng.binomial(df[3].abs(), keep_prob)
    df[3] = downsampled_counts

    # remove regions with 0 signal
    df = df.loc[df[3] != 0]
    df.sort_values([0, 1], inplace=True)
    df.to_csv(save_to, sep="\t", header=False, index=False)


def downsample_bw(in_bw: str, output_prefix: str, chrom_size: str,
                  keep_prob: float, random_generator, use_poisson: bool = False) -> str:
    int_bedgraph = output_prefix + ".bedGraph"
    bigwig_to_bedgraph(in_bw, int_bedgraph)

    # now downsample the file
    sampled_bedgraph = output_prefix + ".sampled.bedGraph"
    keep_prob = min(max(0., keep_prob), 1.)
    downsample_bedgraph(
        int_bedgraph,
        keep_prob=keep_prob,
        save_to=sampled_bedgraph,
        random_generator=random_generator,
        use_poisson=use_poisson,
    )

    # convert the sampled bedGraph to bigWig
    output_bw_file = output_prefix + ".bw"
    bedgraph_to_bigwig(sampled_bedgraph, output_bw_file, chrom_size)

    # clean up
    os.remove(int_bedgraph)
    os.remove(sampled_bedgraph)
    return output_bw_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample multiple bigwig files. "
                                                 "This script should only be used when signals in the "
                                                 "input bigWig files are from the ends of reads.")
    parser.add_argument("-i", "--input", nargs="+")
    parser.add_argument("-s", "--scale-factors", nargs="+", type=float)
    parser.add_argument("-r", "--random-seed", type=int)
    parser.add_argument("-o", "--save-to")
    parser.add_argument("-c", "--chrom-size")

    args = parser.parse_args()

    random_g = np.random.default_rng(args.random_seed)
    for i, bw in enumerate(args.input):
        # first convert to bedGraph
        output_file_prefix = os.path.join(
            args.save_to,
            os.path.splitext(
                os.path.split(bw)[1]  # (dir, filename)
            )[0]  # (root, ext)
        )
        downsample_bw(bw, output_file_prefix, chrom_size=args.chrom_size,
                      keep_prob=args.scale_factors[i], random_generator=random_g)
