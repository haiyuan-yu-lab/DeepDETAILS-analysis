import os
import pybedtools
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from itertools import combinations
from typing import Union, Optional
from torch import nn
from torch.utils.data import DataLoader
from deepdetails.data import SequenceSignalDataset
from deepdetails.helper.utils import calc_counts_per_locus


class ReducedDataset(torch.utils.data.Dataset):
    """
    Raw dataset may have overlap regions
    ReducedDataset as the name suggests, reduces the overlap regions
    by selecting a subset regions. If training regions A, B and C overlap,
    regions A and C will be kept (always the odd indexes)
    """
    def __init__(self, base_data: SequenceSignalDataset,
                 chrom_subset: Optional[tuple] = None,
                 subset: Optional[int] = None,
                 close_threshold: int = 500):
        self.base_data = base_data
        # input regions:
        #    index     0       1       2  3
        # 0      8  chr1  113521  117617  1
        # 1     46  chr1  603470  607566  1
        # 2     51  chr1  627724  631820  1
        # 3     52  chr1  628100  632196  1
        # 4     53  chr1  631971  636067  1
        self.raw_regions = base_data.df.copy()
        if chrom_subset is not None:
            self.raw_regions = self.raw_regions.loc[self.raw_regions[0].isin(chrom_subset)]
        self.raw_regions[3] = self.raw_regions.index.values
        mids = self.raw_regions[[1, 2]].mean(axis=1).astype(int)
        tmp_df = pd.DataFrame({0: self.raw_regions[0], 1: mids, 2: mids + 1, 3: self.raw_regions.index.values})
        tmp_bed = pybedtools.BedTool.from_dataframe(tmp_df).merge(d=close_threshold, c=4, o="distinct")
        reduced_regions = tmp_bed.to_dataframe(disable_auto_names=True, header=None)
        self.final_data_indexes = []
        for candidates in reduced_regions[3].str.split(",").values:
            self.final_data_indexes.extend(candidates[0::2])
        if subset is not None and len(self.final_data_indexes) > subset:
            self.final_data_indexes = np.random.choice(self.final_data_indexes, subset, replace=False).tolist()
        self.base_data.df.iloc[self.final_data_indexes][
            [0, 1, 2, 3, "index"]].to_csv("region_mapping.bed", sep="\t", index=False, header=False)

    @property
    def t_y(self):
        return self.base_data.t_y

    @property
    def t_x(self):
        return self.base_data.t_x

    @property
    def n_clusters(self):
        return self.base_data.n_clusters

    def __getitem__(self, index):
        return self.base_data[int(self.final_data_indexes[index])]

    def __len__(self):
        return len(self.final_data_indexes)


class ModelWithSummarization(pl.LightningModule):
    def __init__(self, base_model: Union[nn.Module, pl.LightningModule],
                 summarizer: str = "weighted_sum", contrast: Optional[str] = None,
                 sample_in_first_dim: bool = False, apply_loads_trick: bool = False):
        """Wrapper model for applying summarization to the predictions

        Parameters
        ----------
        base_model : Union[nn.Module, pl.LightningModule]
            Base model class
        summarizer : str
            Summarization method. Currently, supports "weighted_sum", "sum", and "loads"
        contrast : Optional[str]
            set values such as fc (fold change) or lfc (log fold change) to calculate the contrast between
            the predicted clusters
        sample_in_first_dim : bool, optional
            If True, the first two dimensions of the output from the base model will be swapped.
        apply_loads_trick : bool, optional
            Loads may not be included in the computational graph when using second pass model or models without
            active scale functions. In these cases, captum raises errors like "RuntimeError: One of the differentiated
            Tensors appears to not have been used in the graph."
            Set this to True to forcefully attach loads to the graph to address the issue above.
        """
        super(ModelWithSummarization, self).__init__()
        self.summarizer = summarizer
        self.model = base_model
        self.expected_clusters = base_model.expected_clusters
        self.fc = True if contrast is not None and contrast.upper() == "FC" else False
        self.lfc = True if contrast is not None and contrast.upper() == "LFC" else False
        self.ld = True if contrast is not None and contrast.upper() == "LOAD" else False
        self.sample_in_first_dim = sample_in_first_dim
        self.apply_loads_trick = apply_loads_trick
        self._comp_groups = tuple(combinations(np.arange(base_model.expected_clusters), 2))

        if self.fc or self.lfc:
            print("Overriding summarizer clusters with the following contrast groups")
            for i, g in enumerate(self._comp_groups):
                print(i, g)
            self.expected_clusters = len(self._comp_groups)

    def forward(self, seq: torch.Tensor, atac: torch.Tensor, loads: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        seq : torch.Tensor
            one-hot encoded sequences
        atac : torch.Tensor

        loads : torch.Tensor


        Returns
        -------
        out : torch.Tensor
            shape: batch_size x expected_clusters
        """
        model_outs = self.model([seq, atac], loads)
        pc_profiles, pc_counts, pred_loads = model_outs[:3]
        if self.apply_loads_trick:
            trick = loads.sum(axis=-1).mean()
            pc_counts = [c * trick for c in pc_counts]
        if pc_profiles is not None:
            if isinstance(pc_profiles, torch.Tensor):  # hack for pure PROcapNet
                cluster_preds = torch.exp(self.model.model.log_softmax(pc_profiles)) * torch.exp(pc_counts)[..., None]
                cluster_preds = cluster_preds[None, ...]
            else:
                cluster_preds = calc_counts_per_locus(pc_profiles, pc_counts, True)  # c, m, s, l
            if self.summarizer == "weighted_sum":
                weights = torch.softmax(cluster_preds, axis=-1).detach()
                out = (weights * cluster_preds).sum(axis=-1).sum(axis=-1)
            elif self.summarizer == "weighted_sum_strandless":
                logits = cluster_preds.reshape(cluster_preds.shape[0], cluster_preds.shape[1], -1)
                mean_norm_logits = logits - torch.mean(logits, axis=-1, keepdims=True)
                softmax_probs = torch.nn.Softmax(dim=-1)(mean_norm_logits.detach())
                out = (mean_norm_logits * softmax_probs).sum(axis=-1)
            elif self.summarizer == "sum":
                out = cluster_preds.sum(axis=-1).sum(axis=-1)
            elif self.summarizer == "sum-alone":
                if isinstance(pc_profiles, torch.Tensor):
                    out = pc_counts[None, ...]
                else:
                    out = torch.stack(pc_counts).sum(axis=-1)
            elif self.summarizer == "softmax-counts":
                if isinstance(pc_profiles, torch.Tensor):
                    raise NotImplementedError()
                else:
                    out = torch.softmax(torch.stack(pc_counts).sum(axis=-1), dim=0)
            elif self.summarizer == "loads":
                out = pred_loads
            else:
                raise ValueError(f"{self.summarizer} is not supported.")

            if self.sample_in_first_dim and self.summarizer != "loads":
                out = torch.swapaxes(out, 0, 1)
        else:
            out = pred_loads
        # shape of out: batch_size, n_clusters
        if self.fc or self.lfc:
            out = out + 10e-16
            contrast_out = torch.zeros(seq.shape[0], len(self._comp_groups))
            for i, (gx, gy) in enumerate(self._comp_groups):
                contrast_out[:, i] = out[:, gx] / out[:, gy]

            if self.lfc:
                contrast_out = torch.log2(contrast_out)
            out = contrast_out
        elif self.ld:
            out = out + 10e-16
            out = out / out.sum(axis=-1)[:, None]

        return out


def flush(onehot_seq_encodings: np.ndarray,
          n_clusters: int, contrib_scores: np.ndarray,
          atac_contrib_scores: Optional[np.ndarray] = None,
          load_contrib_scores: Optional[np.ndarray] = None,
          save_to: str = ".", abs_transform: bool = False):
    """
    Flush results/caches into npz files

    Parameters
    ----------
    onehot_seq_encodings : np.ndarray

    n_clusters : int

    contrib_scores : np.ndarray

    atac_contrib_scores : Optional[np.ndarray]

    load_contrib_scores : Optional[np.ndarray]

    save_to : str

    abs_transform : Optional[bool]
        Apply absolute transformation? Turned off by default.

    Returns
    -------

    """
    # save sequence encoding
    np.savez_compressed(os.path.join(save_to, "ohe.npz"), onehot_seq_encodings)

    # save cluster-specific contribution scores
    for c in range(n_clusters):
        np.savez_compressed(os.path.join(save_to, f"C{c}.npz"),
                            contrib_scores[:, c, :, :] if not abs_transform else np.abs(contrib_scores[:, c, :, :]))
    if atac_contrib_scores is not None:
        np.savez_compressed(os.path.join(save_to, f"atac.npz"), atac_contrib_scores,
                            atac_contrib_scores if not abs_transform else np.abs(atac_contrib_scores))
    if load_contrib_scores is not None:
        np.savez_compressed(os.path.join(save_to, f"loads.npz"),
                            load_contrib_scores if not abs_transform else np.abs(load_contrib_scores))
