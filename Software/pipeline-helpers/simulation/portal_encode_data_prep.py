import argparse
import os
from pythonase.run import run_command
from utils import bed_to_cov_bw
try:
    from deepdetails.protocols import prepare_dataset
except ImportError:
    from details.protocols import prepare_dataset


def acc_to_url(accession, ext="bam"):
    return f"https://www.encodeproject.org/files/{accession}/@@download/{accession}.{ext}"


def must_success_run(command):
    stdout, stderr, rc = run_command(command)
    assert rc == 0, (stdout, stderr)


def download_file(in_url, out_file):
    must_success_run(f"wget -qO {out_file} {in_url}")


def merge_bam(in_bams, out_bam):
    assert isinstance(in_bams, list) or isinstance(in_bams, tuple)
    assert len(in_bams) > 1, ""
    must_success_run(f"samtools merge -o {out_bam} {' '.join(in_bams)}")


def bam_to_tagalign(in_bam, layout):
    if layout == "PE":
        must_success_run(f"samtools sort -n {in_bam} -o sorted.bam")
        must_success_run(
            "LC_COLLATE=C bedtools bamtobed -bedpe -mate1 -i sorted.bam | sort -k1,1 -k2,2n -k3,3n | gzip -nc > align.bedpe.gz")
        must_success_run("""zcat -f align.bedpe.gz | awk 'BEGIN{OFS="\t"}{printf "%s\t%s\t%s\tN\t1000\t%s\n%s\t%s\t%s\tN\t1000\t%s\n",$1,$2,$3,$9,$4,$5,$6,$10}' | gzip -nc > tagAlign.gz""")
        must_success_run("rm sorted.bam")
        must_success_run("rm align.bedpe.gz")
    elif layout == "SE":
        cmd = """bedtools bamtobed -i %s | awk 'BEGIN{OFS="\t"}{$4="N";$5="1000";print $0}' | gzip -nc > tagAlign.gz;""" % in_bam
        must_success_run(cmd)
    else:
        raise ValueError(f"Unknown layout {layout}")

    assert os.path.exists("tagAlign.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bams", required=True, nargs="+", help="Bam accessions")
    parser.add_argument("-p", "--peaks", required=True, help="Peak accession")
    parser.add_argument("-f", "--fragments", required=True, help="Fragments file")
    parser.add_argument("-m", "--barcode", required=True, help="Cell barcode to cell type mapping")
    parser.add_argument("-c", "--chrom-size", required=True, help="Chromosome size")
    parser.add_argument("--genome-fa", type=str, required=True)
    parser.add_argument("-l", "--layout", help="Library type: PE or SE")
    parser.add_argument("--accessible-regions", dest="accessible_peaks", required=False)
    parser.add_argument("--skip-preflight", required=False, action="store_true")
    parser.add_argument("--keep-frags", required=False, action="store_true")

    parser.add_argument(
        "--merge-overlap-peaks", type=int, default=0,
        help="Minimum overlap between features allowed for features to be merged. ")
    parser.add_argument(
        "--target-sliding-sum", default=0, type=int,
        help="Apply sliding sum to the target signals if the value is greater than 0.")
    parser.add_argument("-r", "--random-seed", type=int, default=1234567)
    parser.add_argument("--background-blacklist", type=str, required=False)

    args = parser.parse_args()

    # get the peak file
    download_file(acc_to_url(args.peaks, "bed.gz"), "peaks.bed.gz")

    # get the bam files, merge replicates if needed
    if len(args.bams) > 1:
        in_bams = []
        for i, bam_acc in enumerate(args.bams):
            url = acc_to_url(bam_acc, "bam")
            download_file(url, f"b{i}.bam")
            in_bams.append(f"b{i}.bam")
        merge_bam(in_bams, "out.bam")

        for i in range(len(args.bams)):
            must_success_run(f"rm b{i}.bam")
    else:
        download_file(acc_to_url(args.bams[0], "bam"), "out.bam")

    # convert bam file to tagAlign
    bam_to_tagalign("out.bam", args.layout)
    must_success_run("rm out.bam")

    # convert tagAlign to bigwig
    bed_to_cov_bw("tagAlign.gz", "out.pl.bw", args.chrom_size, limit_strand_to="+")
    bed_to_cov_bw("tagAlign.gz", "out.mn.bw", args.chrom_size, limit_strand_to="-")

    accessible_peaks = None
    if args.accessible_peaks and os.path.exists(args.accessible_peaks):
        accessible_peaks = args.accessible_peaks
    else:
        fragment_parent = os.path.split(args.fragments)[0]
        expected_peak_file = os.path.join(fragment_parent, "atacPeaks/macs2_peaks.narrowPeak")

        if os.path.exists(expected_peak_file):
            accessible_peaks = expected_peak_file

    # build dataset
    prepare_dataset(**{
        "regions": ("peaks.bed.gz", ),
        "bulk_pl": "out.pl.bw",
        "bulk_mn": "out.mn.bw",
        "fragments": args.fragments,
        "barcodes": args.barcode,
        "chrom_size": args.chrom_size,
        "accessible_peaks": accessible_peaks,
        "save_to": ".",
        "window_size": 4096,
        "background_sampling_ratio": 1.,
        "final_regions": False,
        "merge_overlap_peaks": args.merge_overlap_peaks,
        "background_blacklist": args.background_blacklist,
        "target_sliding_sum": args.target_sliding_sum,
        "genome_fa": args.genome_fa,
        "seed": args.random_seed,
        "skip_preflight": args.skip_preflight,
        "keep_frags": args.keep_frags,
    })
