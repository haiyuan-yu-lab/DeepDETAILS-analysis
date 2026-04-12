"""Merge multi-sample scATAC fragment and barcode tables with per-sample suffixes.

Expects 10x-style fragment TSVs (chrom, start, end, barcode, …) and single-column
barcode lists. Optionally concatenates MACS2 narrowPeak files from a fixed layout
next to each fragment path.
"""
from __future__ import annotations

import argparse
import os.path

import pandas as pd
import pybedtools

# Fragment files: at least chrom, start, end, barcode (0-based column index 3).
_MIN_FRAGMENT_COLUMNS = 4
_MIN_BARCODE_COLUMNS = 1


def _validate_fragments_table(df: pd.DataFrame, path: str) -> None:
    """Require non-empty fragments with chrom, numeric start/end, and barcode column."""
    ncols = df.shape[1]
    if ncols < _MIN_FRAGMENT_COLUMNS:
        raise ValueError(
            f"Fragment file {path!r} must have at least {_MIN_FRAGMENT_COLUMNS} "
            f"tab-separated columns (chrom, start, end, barcode); got {ncols}."
        )
    if df.empty:
        raise ValueError(f"Fragment file {path!r} has no rows.")
    if df.iloc[:, : _MIN_FRAGMENT_COLUMNS].isna().any().any():
        raise ValueError(
            f"Fragment file {path!r} has missing values in the first {_MIN_FRAGMENT_COLUMNS} columns."
        )
    try:
        pd.to_numeric(df.iloc[:, 1], errors="raise")
        pd.to_numeric(df.iloc[:, 2], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Fragment file {path!r}: columns 2 and 3 (1-based) must be numeric start and end positions."
        ) from exc


def _validate_barcode_table(df: pd.DataFrame, path: str) -> None:
    """Require non-empty single-column (or wider) barcode list; first column is the barcode."""
    ncols = df.shape[1]
    if ncols < _MIN_BARCODE_COLUMNS:
        raise ValueError(
            f"Barcode file {path!r} must have at least {_MIN_BARCODE_COLUMNS} "
            f"tab-separated column(s); got {ncols}."
        )
    if df.empty:
        raise ValueError(f"Barcode file {path!r} has no rows.")
    if df.iloc[:, 0].isna().any():
        raise ValueError(f"Barcode file {path!r} has missing values in the first column.")


def main(in_frags: list[str], in_barcodes: list[str], save_to: str) -> None:
    """Suffix barcodes and fragment cell IDs per sample index, then write combined TSV.gz files.

    For sample index ``i``, appends ``-{i}`` to barcode file column 0 and fragment
    file column 3. Writes ``fragments.tsv.gz`` and ``barcodes.tsv.gz`` under ``save_to``.

    Parameters
    ----------
    in_frags
        Paths to fragment TSVs (same order as ``in_barcodes``).
    in_barcodes
        Paths to barcode TSVs (one barcode per line in the first column).
    save_to
        Output directory.
    """
    if len(in_frags) != len(in_barcodes):
        raise ValueError(
            f"Got {len(in_frags)} fragment file(s) but {len(in_barcodes)} barcode file(s); "
            "counts must match."
        )

    barcode_dfs: list[pd.DataFrame] = []
    frag_dfs: list[pd.DataFrame] = []

    for i, (in_frag_file, in_barcode_file) in enumerate(zip(in_frags, in_barcodes)):
        barcodes = pd.read_csv(in_barcode_file, header=None, sep="\t")
        _validate_barcode_table(barcodes, in_barcode_file)
        barcodes[0] = barcodes[0] + f"-{i}"
        barcode_dfs.append(barcodes)

        frags = pd.read_csv(in_frag_file, header=None, sep="\t")
        _validate_fragments_table(frags, in_frag_file)
        frags[3] = frags[3] + f"-{i}"
        frag_dfs.append(frags)

    pd.concat(frag_dfs, ignore_index=True).sort_values([0, 1]).to_csv(
        os.path.join(save_to, "fragments.tsv.gz"),
        index=False,
        header=False,
        sep="\t",
    )
    pd.concat(barcode_dfs, ignore_index=True).to_csv(
        os.path.join(save_to, "barcodes.tsv.gz"),
        index=False,
        header=False,
        sep="\t",
    )


def supplementary_main(in_frags: list[str], save_to: str) -> None:
    """Concatenate ``atacPeaks/macs2_peaks.narrowPeak`` next to each fragment path into one BED.

    For each path ``f`` in ``in_frags``, reads ``{dirname(f)}/atacPeaks/macs2_peaks.narrowPeak``,
    concatenates with pybedtools, and saves under ``save_to/atacPeaks/``.

    Parameters
    ----------
    in_frags
        Fragment file paths (used only to infer sibling peak file locations).
    save_to
        Output root; peaks are written to ``save_to/atacPeaks/macs2_peaks.narrowPeak``.
    """
    peak_files = [
        os.path.join(os.path.split(f)[0], "atacPeaks/macs2_peaks.narrowPeak") for f in in_frags
    ]
    peak_beds = [pybedtools.BedTool(f) for f in peak_files]
    merged_peaks = pybedtools.BedTool.cat(*peak_beds)
    parent_dir = os.path.join(save_to, "atacPeaks")
    os.makedirs(parent_dir, exist_ok=True)
    merged_peaks.saveas(os.path.join(parent_dir, "macs2_peaks.narrowPeak"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine scATAC fragment and barcode tables across samples.",
    )
    parser.add_argument("-f", "--in-frags", type=str, nargs="+", required=True)
    parser.add_argument("-b", "--in-barcodes", type=str, nargs="+", required=True)
    parser.add_argument("-s", "--save-to", type=str, default=".")
    args = parser.parse_args()

    main(args.in_frags, args.in_barcodes, args.save_to)

    supplementary_main(args.in_frags, args.save_to)
