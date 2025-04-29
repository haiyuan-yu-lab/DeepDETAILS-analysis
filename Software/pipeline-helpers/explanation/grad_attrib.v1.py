import argparse
import os
import torch
import pytorch_lightning as pl
import numpy as np
from typing import Union
from torch.utils.data import DataLoader
from tqdm import tqdm
from captum.attr import IntegratedGradients, InputXGradient, Saliency, DeepLiftShap
from pythonase.print_functions import message_with_time
from deepdetails.model.wrapper import DeepDETAILS
from deepdetails.data import SequenceSignalDataset
from summarization_v1 import ModelWithSummarization, ReducedDataset, flush


def input_x_gradient(model: Union[torch.nn.Module, pl.LightningModule],
                     dataset: torch.utils.data.Dataset, batch_size: int,
                     save_to: str = ".", abs_transform: bool = False):
    """
    Input x Gradient

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]

    dataset : torch.utils.data.Dataset

    batch_size : int

    save_to : str

    abs_transform : bool, optional


    Returns
    -------

    """
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    n_samples = len(dataset)
    n_batches = len(data_iter)
    seq_len = dataset.t_x
    n_clusters = model.expected_clusters
    ixg = InputXGradient(model)
    device = model.device

    # The one-hot encoded sequences and attributions are assumed to be in length-last format,
    # i.e., have the shape (# examples, 4, sequence length).

    # this denotes the identity of the sequence
    onehot_seq_encodings = np.zeros((n_samples, 4, seq_len))
    # attribution matrix
    contrib_scores = np.zeros((n_samples, n_clusters, 4, seq_len))
    atac_contrib_scores = np.zeros((n_samples, n_clusters, seq_len))
    loads_contrib_scores = np.zeros((n_samples, n_clusters))

    for batch_idx, datum in enumerate(tqdm(data_iter, disable="SLURM_JOB_ID" in os.environ)):
        message_with_time(f"Working on batch {batch_idx} / {n_batches}")
        sequence = datum[0][0].squeeze()  # acgt (4), sequence_length
        absolute_start_coord = batch_idx * batch_size
        absolute_end_coord = min((batch_idx + 1) * batch_size, n_samples)
        onehot_seq_encodings[absolute_start_coord:absolute_end_coord, :, :] = sequence
        for target in range(n_clusters):
            # It is assumed that for all given input tensors, dimension 0 corresponds to
            # the number of examples, and if multiple input tensors are provided,
            # the examples must be aligned appropriately.
            attributions = ixg.attribute(
                inputs=(datum[0][0].to(device), datum[0][1].to(device), datum[4].to(device)),
                target=target)

            # attributions will always be the same size as the provided inputs,
            # with each value providing the attribution of the corresponding input index.
            # If a single tensor is provided as inputs, a single tensor is returned.
            # If a tuple is provided for inputs, a tuple of corresponding sized tensors is returned.
            contrib_scores[absolute_start_coord:absolute_end_coord, target, :, :] = attributions[
                0].detach().cpu()
            atac_contrib_scores[absolute_start_coord:absolute_end_coord, target, :] = attributions[1].detach().cpu()[:,
                                                                                      target, :]
            loads_contrib_scores[absolute_start_coord:absolute_end_coord, target] = attributions[2].detach().cpu()[:,
                                                                                    target]
    flush(
        onehot_seq_encodings=onehot_seq_encodings,
        contrib_scores=contrib_scores,
        atac_contrib_scores=atac_contrib_scores,
        load_contrib_scores=loads_contrib_scores,
        n_clusters=n_clusters, save_to=save_to, abs_transform=abs_transform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("-o", "--save-to", default=".")
    parser.add_argument("--chromosome-subset", action="store", dest="cs", nargs="*", required=False,
                        help="Chromosomes that will be used as the validation set")
    parser.add_argument("--region-subset", action="store", dest="region_subset", type=int, required=False,
                        help="Sampling region subset for attribution calculation")
    parser.add_argument("--all-regions", action="store_false", dest="pos_only",
                        help="Use all instead of just non-background regions")
    parser.add_argument("--attr-method", action="store", type=str, default="ixg", required=False,
                        choices=("ixg", "inputxgrad", ), )
    parser.add_argument("--abs-attr", action="store_true",
                        help="Calculate the absolute attribution as done in Enformer")
    parser.add_argument("--contrast", action="store", type=str, required=False,
                        choices=("fc", "lfc", "load"), help="Use contrast (foldchange or log foldchange) between clusters "
                                                    "instead of vanilla output")
    parser.add_argument("--data-mode", action="store", dest="data_mode", type=str, default="full", required=False,
                        choices=("full", "mean", "max", "min", "per_bin_mean", "per_bin_max", "per_bin_min"), )
    parser.add_argument("--bins", action="store", dest="bins",
                        type=int, default=0, required=False)
    parser.add_argument("--y-length", default=1000, type=int)
    parser.add_argument("--batch-size", action="store", help="Batch size",
                        dest="batch_size", type=int, default=32, required=False)
    parser.add_argument("--data-workers", action="store", help="Number of workers for data prep",
                        dest="num_workers", type=int, default=16, required=False)
    parser.add_argument("--model-file")
    parser.add_argument("--disable-scale-func", action="store_true")
    parser.add_argument("--disable-teacher-forcing", dest="teacher_forcing", action="store_false")
    parser.add_argument("-s", "--prediction-summarization", default="weighted_sum")
    parser.add_argument("--apply-trick-for-loads", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--region-close-thred", action="store", type=int, default=500,
                        help="Regions close to each other (distance as defined by this parameter) will be reduced.")

    args = parser.parse_args()

    raw_ds = SequenceSignalDataset(
        root=args.root, y_length=args.y_length, is_training=-1, chromosomal_val=None, chromosomal_test=None,
        non_background_only=args.pos_only,
    )
    full_ds = SequenceSignalDataset(
        root=args.root, y_length=args.y_length, is_training=-1,  chromosomal_val=None, chromosomal_test=None
    )

    test_ds = ReducedDataset(raw_ds, chrom_subset=args.cs,
                             close_threshold=args.region_close_thred,
                             subset=args.region_subset)

    if args.disable_scale_func:
        base_model = DeepDETAILS.load_from_checkpoint(args.model_file, scale_function_placement="disable")
    else:
        base_model = DeepDETAILS.load_from_checkpoint(args.model_file)

    if not args.teacher_forcing:
        disabled = False
        try:
            base_model.model.teacher_forcing = False
            disabled = True
        except AttributeError:
            pass
        try:
            base_model.teacher_forcing = False
            disabled = True
        except AttributeError:
            pass
        if not disabled:
            print("--disable-teacher-forcing is not effective because the model doesn't support for this attr")

    model = ModelWithSummarization(
        base_model=base_model,
        summarizer=args.prediction_summarization, contrast=args.contrast,
        sample_in_first_dim=True, apply_loads_trick=args.apply_trick_for_loads
    ).to(args.device)

    if args.attr_method == "inputxgrad" or args.attr_method == "ixg":
        input_x_gradient(model, test_ds, args.batch_size, args.save_to, abs_transform=args.abs_attr)
    else:
        raise ValueError("Unsupported attr method: {}".format(args.attr_method))
