import argparse
import os.path

import pandas as pd
from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_label")

    args = parser.parse_args()

    row_mapping = pd.read_csv("row_mapping.csv", index_col=0).set_index("mapped").to_dict()["original"]

    # remap predictions from the group mode to the original space
    frac_file = None
    # the estimated fractions will be stored in:

    _adj_frac_file = f"CIBERSORTxGEP_{args.run_label}_Fractions-Adjusted.txt"
    _frac_file = f"CIBERSORTxGEP_{args.run_label}_Fractions.txt"
    if os.path.exists(_adj_frac_file):
        # CIBERSORTxGEP_*_Fractions-Adjusted.txt, if batch effect correction is enabled
        frac_file = _adj_frac_file
    elif os.path.exists(_frac_file):
        # CIBERSORTxGEP_*_Fractions.txt, otherwise
        frac_file = _frac_file
    else:
        raise IOError("Expecting a file for fraction estimation")

    gep_frac = pd.read_csv(frac_file, sep="\t", index_col="Mixture")
    gep_out = pd.read_csv(f"CIBERSORTxGEP_{args.run_label}_GEPs.txt", sep="\t", index_col=0)
    frac_sub = gep_frac[gep_out.columns]
    gep_out.index = gep_out.index.map(row_mapping)
    gep_out.to_csv("GEPs.csv")
    for ds, frac in frac_sub.iterrows():
        (gep_out * frac).to_csv(f"GEPs.{ds}.csv")

    # remap predictions from the highres mode to the original space
    per_sample_ct_exp_dicts = {}
    for cell_type in gep_out.columns:
        _expected_file = glob(f"CIBERSORTxHiRes_{args.run_label}_{cell_type}_Window*.txt")[0]
        _hires_out = pd.read_csv(_expected_file, sep="\t", index_col=0)

        for sample in _hires_out.columns:
            if sample not in per_sample_ct_exp_dicts:
                per_sample_ct_exp_dicts[sample] = {}
            per_sample_ct_exp_dicts[sample][cell_type] = _hires_out[sample]

    for sample in per_sample_ct_exp_dicts:
        _df = pd.DataFrame(per_sample_ct_exp_dicts[sample])
        _df.index = _df.index.map(row_mapping)
        _df.to_csv(f"{sample}.hires.csv")

        (_df * frac_sub.loc[sample]).to_csv(f"{sample}.scaled.hires.csv")
