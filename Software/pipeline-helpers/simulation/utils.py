import argparse
import os
from typing import Optional
from pythonase.run import run_command


def bedgraph_to_bigwig(in_bedgraph_path: str, out_bigwig_path: str, chrom_size_path: str):
    """
    Convert a file in bedGraph format to bigWig format

    Parameters
    ----------
    in_bedgraph_path : str

    out_bigwig_path : str

    chrom_size_path : str


    Returns
    -------

    """
    # get chromosomes that have size info
    allowed_chromosomes = set()
    with open(chrom_size_path, "r") as csf:
        for line in csf:
            allowed_chromosomes.add(line.strip().split()[0])

    # filter bedGraph and write to output file
    tmp_file = f"{in_bedgraph_path}.1"
    with open(in_bedgraph_path, "r") as input_file, open(tmp_file, "w") as output_file:
        for line in input_file:
            parts = line.split("\t")
            chromosome = parts[0]
            if chromosome in allowed_chromosomes:
                output_file.write(line)

    # convert the filtered bedGraph file into bigWig format
    cmd = f"bedGraphToBigWig {tmp_file} {chrom_size_path} {out_bigwig_path}"
    run_command(cmd, raise_exception=True)

    # remove the temporary file
    os.remove(tmp_file)


def bed_to_cov_bw(in_bed_path: str, out_bigwig_path: str, chrom_size_path: str,
                  rpm_norm: Optional[int] = None, report_5p_cov: bool = False,
                  limit_strand_to: Optional[str] = None):
    """
    Convert a file in bed format to bigWig format (coverage)

    Parameters
    ----------
    in_bed_path : str

    out_bigwig_path : str

    chrom_size_path : str

    rpm_norm : Optional[int]
        Give the total read counts to enable RPM normalization
    report_5p_cov : bool
        Set it as True if you only want 5' signal
    limit_strand_to : Optional[str]
        +: forward strand
        -: reverse strand
        None: no limitations

    Returns
    -------

    """
    scalar = 1. if rpm_norm is None else 1000 * 1000 / rpm_norm
    is_p5 = "-5" if report_5p_cov else ""
    strand = f"-strand {limit_strand_to}" if limit_strand_to is not None else ""
    cmd = f"bedtools genomecov -i {in_bed_path} -g {chrom_size_path} -bg -scale {scalar} {is_p5} {strand} > {out_bigwig_path}.bg"
    try:
        run_command(cmd, raise_exception=True)
    except RuntimeError:
        new_cmd = 'awk BEGIN{OFS="\t";FS="\t"}{print $1,0,$2} %s | ' % chrom_size_path
        new_cmd += f"bedtools intersect -a {in_bed_path} -b stdin -u | "
        new_cmd += f"bedtools genomecov -i stdin -g {chrom_size_path} -bg -scale {scalar} {is_p5} {strand} > {out_bigwig_path}.bg"
        run_command(new_cmd, raise_exception=True)

    bedgraph_to_bigwig(f"{out_bigwig_path}.bg", out_bigwig_path, chrom_size_path)
    os.remove(f"{out_bigwig_path}.bg")


def bigwig_to_bedgraph(in_bigwig_path: str, out_bedgraph_path: str):
    """
    Convert a file in bigWig format to bedGraph format

    Parameters
    ----------
    in_bigwig_path : str

    out_bedgraph_path : str


    Returns
    -------

    """
    cmd = f"bigWigToBedGraph {in_bigwig_path} {out_bedgraph_path}"
    run_command(cmd, raise_exception=True)


def main():
    parser = argparse.ArgumentParser(description="A simple calculator CLI program.")

    # Subparsers for subcommands
    subparsers = parser.add_subparsers(help="Available subcommands")

    # Subcommand: bed_to_cov_bw
    parser_add = subparsers.add_parser(
        "bed_to_cov_bw",
        help="Convert reads in bed format to coverage tracks in bigWig format")
    parser_add.add_argument("-i", "--in-bed-path", type=str, required=True,)
    parser_add.add_argument("-o", "--out-bigwig-path", type=str, required=True,)
    parser_add.add_argument("-c", "--chrom-size-path", type=str, required=True,)
    parser_add.add_argument("-r", "--rpm-norm", type=float, default=None)
    parser_add.add_argument("-5", "--report-5p-cov", action="store_true")
    parser_add.add_argument("-s", "--limit-strand-to", type=str,
                            choices=["+", "-", None], required=False, default=None)
    parser_add.set_defaults(func=bed_to_cov_bw)

    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    real_args = vars(parser.parse_args())
    real_args.pop("func")
    args.func(**real_args)


if __name__ == "__main__":
    main()
