import argparse
import os
import torch
import torchmetrics
import wandb
import numpy as np
import pandas as pd
from einops import rearrange
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from pythonase.io import to_csv_with_comments
from deepdetails.data import SequenceSignalDataset
from deepdetails.model.wrapper import DeepDETAILS


def load_checkpoint(checkpoint: str, wandb_project: str = "") -> str:
    if os.path.exists(checkpoint):
        ckpt_file = checkpoint
    else:
        # download checkpoint locally (if not already cached)
        run = wandb.init(project=wandb_project)
        artifact = run.use_artifact(checkpoint, type="model")
        artifact_dir = artifact.download()
        ckpt_file = Path(artifact_dir) / "model.ckpt"
    return ckpt_file


def arg_builder() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("save_to", help="Prefix to the outputs")
    group = parser.add_argument_group("Evaluation functional")
    group.add_argument("-c", "--checkpoints", nargs="+", required=True,
                       help="Model checkpoint file or version info on the cloud")

    group = parser.add_argument_group("Data and Dataloader")
    group.add_argument(
        "--root", help="Data directory", required=True)
    group.add_argument("--pos-only", action="store_true", dest="pos_only",
                       help="Only use non-background regions")
    group.add_argument("--chromosomal-validation", "--cv", action="store", dest="cv", default=("chr22",),
                       help="Chromosomes that will be used as the validation set",
                       required=False, nargs="*")
    group.add_argument("--chromosomal-testing", "--ct", action="store", dest="ct", default=("chr19",),
                       help="Chromosomes that will be used as the testing set",
                       required=False, nargs="*")
    group.add_argument("--y-length", default=1000, type=int)
    group.add_argument("--batch-size", action="store", help="Batch size",
                       dest="batch_size", type=int, default=32, required=False)
    group.add_argument("--data-workers", action="store", help="Number of workers for data prep",
                       dest="num_workers", type=int, default=16, required=False)

    group = parser.add_argument_group("Running configuration")
    group.add_argument("-l", "--labels", dest="cluster_labels", nargs="*", type=str, required=False)
    group.add_argument("--no-chrom-cv", action="store_false", dest="chrom_cv", required=False,
                       help="Set this switch if you want to evaluate on all chromosomes")
    group.add_argument("--gpus", action="store", dest="gpus", nargs="*",
                       type=int, default=(0,), required=False)
    group.add_argument("--hide-progress-bar", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_builder()

    n_models = len(args.checkpoints)

    if args.chrom_cv:
        test_ds = SequenceSignalDataset(
            root=args.root, y_length=args.y_length, is_training=2,
            chromosomal_val=args.cv, chromosomal_test=args.ct, non_background_only=args.pos_only,
        )
    else:
        test_ds = SequenceSignalDataset(
            root=args.root, y_length=args.y_length, is_training=-1,
            chromosomal_val=None, chromosomal_test=None, non_background_only=args.pos_only,
        )

    n_clusters = test_ds.n_clusters
    test_iter = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    try:
        pred_labels = test_ds.cluster_names
    except:
        pred_labels = args.cluster_labels if args.cluster_labels is not None else [f"C{i}" for i in range(n_clusters)]

    device = f"cuda:{args.gpus[0]}"

    model_wrappers = [
        DeepDETAILS.load_from_checkpoint(
            load_checkpoint(ckpt_file), map_location=device
        ).to(device).eval() for ckpt_file in args.checkpoints
    ]

    counts = {model_idx: [] for model_idx in range(n_models)}
    counts["truth"] = []
    final_counts_dfs = {}
    final_sl_counts_dfs = {}
    
    # export predictions
    with torch.no_grad():
        for batch in tqdm(test_iter, disable=args.hide_progress_bar):
            x, expected_counts, _, expected_per_cluster_profiles, loads, _ = batch

            if isinstance(x, list):
                x = [x[0].to(device), x[1].to(device)]
            else:
                x = x.to(device)
            loads = loads.to(device)

            true_counts = rearrange(torch.stack(expected_per_cluster_profiles).sum(axis=-1), "c m s -> m (c s)")
            for l in true_counts:
                counts["truth"].append(l.cpu().numpy().tolist())

            for model_idx in range(n_models):
                mw = model_wrappers[model_idx]
                model_outs = mw.model(x, loads)
                pc_profiles, pc_counts, _ = model_outs[:3]

                ps_counts = torch.stack(pc_counts).sum(axis=0)
                scale_factors = expected_counts.to(ps_counts.device) / ps_counts
                if torch.isinf(scale_factors).sum().item() > 0:
                    scale_factors = torch.nan_to_num(scale_factors, 0.)
                pc_counts = [pcc * scale_factors for pcc in pc_counts]
                flatten_counts = rearrange(torch.stack(pc_counts), "c m s -> m (c s)")
                for l in flatten_counts:
                    counts[model_idx].append(l.cpu().numpy().tolist())
    strand_suffix = ["pl", "mn"]
    counts_cols = [f"{x}_{y}" for x in pred_labels for y in strand_suffix]
    for k, v in counts.items():
        counts_df = pd.DataFrame(v,
                                 index=test_ds.df[0] + ":" + test_ds.df[1].map(str) + "-" + test_ds.df[2].map(str))
        sl_counts_df = pd.DataFrame(
            {label: counts_df[i * 2] + counts_df[i * 2 + 1] for i, label in enumerate(pred_labels)},
            index=counts_df.index, )
        sl_counts_df.to_csv(f"{args.save_to}.m{k}.counts.csv.gz")
        final_counts_dfs[k] = counts_df
    counts_df = pd.DataFrame(counts["truth"],
                             index=test_ds.df[0] + ":" + test_ds.df[1].map(str) + "-" + test_ds.df[2].map(str))
    _c = counts_df.copy()
    _c.columns = counts_cols
    _c.to_csv(args.save_to + ".truth.counts.csv.gz")
    final_counts_dfs["truth"] = counts_df

    for k, v in final_counts_dfs.items():
        if v.shape[1] % 2:
            raise ValueError("Expecting even number of columns (50% for the forward and the other for the reverse)")
        final_sl_counts_dfs[k] = pd.DataFrame({
            f"{i}": v[i * 2] + v[i * 2 + 1] for i in range(n_clusters)
        }, index=v.index, )

    counts_shape_cor_res = []
    counts_rank_cor_res = []
    for model_idx in range(n_models):
        ref_df = final_sl_counts_dfs["truth"].fillna(0.)
        exp_df = final_sl_counts_dfs[model_idx].fillna(0.)

        shared_regions = tuple(
            set(ref_df.index.values).intersection(set(exp_df.index.values)))
        _exp_df = exp_df.loc[exp_df.index.isin(shared_regions)]
        _ref_df = ref_df.loc[ref_df.index.isin(shared_regions)]
        transform_funcs = (lambda _x: _x, np.log1p)
        transform_labels = ("raw", "log")
        for l, f in zip(transform_labels, transform_funcs):
            exp1 = f(_exp_df)
            ref = f(_ref_df)
            assert all([c in ref.columns for c in exp1.columns])

            # shape corr
            for c in ref.columns:
                # columns: model, transform, cluster, corr coef, mse
                counts_shape_cor_res.append((
                    model_idx, l, c,
                    np.corrcoef(exp1[c], ref[c])[0, 1],
                    torchmetrics.functional.concordance_corrcoef(
                        torch.from_numpy(exp1[c].values),
                        torch.from_numpy(ref[c].values)).item(),
                    np.square(exp1[c] - ref[c]).mean()))

            # rank corr
            for r in _ref_df.index:
                # model, region, transform, corr coef, mse
                counts_rank_cor_res.append((
                    model_idx, r, l,
                    np.corrcoef(exp1.loc[r], ref.loc[r])[0, 1],
                    torchmetrics.functional.concordance_corrcoef(
                        torch.from_numpy(exp1.loc[r].values),
                        torch.from_numpy(ref.loc[r].values)).item(),
                    np.square(exp1.loc[r] - ref.loc[r]).mean()))

    to_csv_with_comments(
        pd.DataFrame(counts_shape_cor_res, columns=["Model", "Transform", "Cluster", "Corr coef", "CCC", "MSE"]),
        save_to= f"{args.save_to}.csc.csv.gz", index=False,
        additional_comment_lines=tuple([f"{k}: {v}" for k, v in vars(args).items()]),)
    to_csv_with_comments(
        pd.DataFrame(counts_rank_cor_res, columns=["Model", "Region", "Transform", "Corr coef", "CCC", "MSE"]),
        save_to=f"{args.save_to}.crc.csv.gz", index=False,
        additional_comment_lines=tuple([f"{k}: {v}" for k, v in vars(args).items()]), )
