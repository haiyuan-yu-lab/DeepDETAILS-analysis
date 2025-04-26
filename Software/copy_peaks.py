import argparse
import os
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("peaks")

    args = parser.parse_args()

    biosamples = pd.read_csv("config/biosamples.tsv", sep="\t")

    for b in biosamples["biosample"].values:
        dest = f"results/{b}/Peaks"
        os.makedirs(dest)
        os.symlink(args.peaks, os.path.join(dest, "macs2_peaks.narrowPeak"))
