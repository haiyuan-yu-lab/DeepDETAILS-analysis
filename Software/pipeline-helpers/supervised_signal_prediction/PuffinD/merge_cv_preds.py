import argparse
import os
import pyBigWig
import numpy as np


def main(cell_types, chrom_sizes_filepath, chrom_names):
    with open(chrom_sizes_filepath) as f:
        chrom_sizes_lines = [line.strip().split('\t') for line in f]
        chrom_sizes = [(line[0], int(line[1])) for line in chrom_sizes_lines]
        chrom_sizes_dict = {x[0]: x[1] for x in chrom_sizes}

    sorted_chroms = [c for c in chrom_sizes_dict.keys() if c in chrom_names]

    # save
    for cell_type in cell_types:
        for strand_idx, strand in enumerate(["pl", "mn"]):
            bw_filename = f"{cell_type}.{strand}.bw"

            with pyBigWig.open(bw_filename, "w") as bw:
                bw.addHeader(chrom_sizes)

                for chrom in sorted_chroms:
                    print(f"Working on chrom {chrom} ({strand}, {cell_type})")
                    # e.g. Caco2.mn.chr1.bw
                    expected_file = f"{cell_type}.{strand}.{chrom}.bw"
                    assert os.path.exists(expected_file)

                    with pyBigWig.open(expected_file) as in_bw:
                        n = chrom_sizes_dict[chrom]
                        preds = in_bw.values(chrom, 0, n, numpy=True)
                        starts = np.arange(n)
                        ends = starts + 1

                        bw.addEntries([chrom, ] * n, starts, ends=ends, values=preds)
                    print(f"Finished processing chrom {chrom} ({strand}, {cell_type})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cell-types", nargs="+", required=True, type=str)
    parser.add_argument("--chroms", nargs="+", required=False, type=str,
                        default=("chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
                                 "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18",
                                 "chr19", "chr20", "chr21", "chr22", "chrX"))
    parser.add_argument("--chrom-sizes", dest="chrom_sizes", required=True, type=str)

    args = parser.parse_args()

    main(cell_types=args.cell_types,
         chrom_sizes_filepath=args.chrom_sizes,
         chrom_names=args.chroms)
