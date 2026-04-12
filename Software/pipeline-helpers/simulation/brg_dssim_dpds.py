import argparse
import os
import pandas as pd


def main():
    """
    This script serves as a bridge between build_dataset.py and DeepDETAILS' real-world data processing pipeline

    Returns
    -------

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True, type=str, help="Input dir")
    parser.add_argument("-o", "--output-dir", default=".", type=str, help="Output dir")

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    sc_norm_file = os.path.join(input_dir, "scatac.norm.csv")
    assert os.path.exists(sc_norm_file), sc_norm_file

    sc_norms = pd.read_csv(sc_norm_file, header=None)

    # per cell type scATAC
    pct_scatac_dfs = []
    pct_bcs_dfs = []
    for idx, file in enumerate(sc_norms[0]):
        pct_scatac_frag_file = os.path.join(input_dir, file)
        _, cell_type, _, _ = file.split(".")
        assert os.path.exists(pct_scatac_frag_file), pct_scatac_frag_file

        _fragments = pd.read_csv(pct_scatac_frag_file, sep="\t", header=None)
        _fragments[3] = _fragments[3] + str(idx)
        pct_scatac_dfs.append(_fragments)

        _bcs = pd.DataFrame({
            0: _fragments[3].unique(),
            1: cell_type
        })

        pct_bcs_dfs.append(_bcs)

    merged_frags = pd.concat(pct_scatac_dfs, ignore_index=True).sort_values([0, 1], ignore_index=True)
    merged_frags.to_csv(os.path.join(output_dir, "merged_fragments.tsv.gz"), sep="\t", header=False, index=False)

    merged_bcs = pd.concat(pct_bcs_dfs, ignore_index=True)
    merged_bcs.to_csv(os.path.join(output_dir, "merged_bcs.tsv"), sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
