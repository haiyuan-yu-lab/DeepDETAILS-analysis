import argparse
import os
import numpy as np
import tabix
import torch
import selene_sdk
import pyBigWig
from torch import nn
from matplotlib import pyplot as plt
from selene_sdk.targets import Target
from selene_sdk.samplers import RandomPositionsSampler
from selene_sdk.samplers.dataloader import SamplerDataLoader

torch.set_default_tensor_type('torch.FloatTensor')


class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """

    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None,
                 replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
            self.initialized = True

        wigmat = np.vstack([c.values(chrom, start, end, numpy=True)
                            for c in self.data])  # k, 100_000

        if self.blacklists is not None:
            if self.replacement_indices is None:
                for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(
                        self.blacklists, self.blacklists_indices, self.replacement_indices,
                        self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = wigmat[
                                                                                                replacement_indices,
                                                                                                np.fmax(int(s) - start,
                                                                                                        0): int(
                                                                                                    e) - start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        wigmat = np.abs(wigmat)
        # > the base pair–resolution count profiles were averaged after applying log10 (x + 1) transformation,
        # > where x is the read count, with plus and minus strand profiles aggregated separately
        # since our input bigwig files are in the original scale, we add the required transformation here
        wigmat = np.log10(1 + wigmat)
        return wigmat


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2, fused=True):
        super(ConvBlock, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, 1, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        return x + self.conv(x)


class PuffinD(nn.Module):
    def __init__(self):
        """
        Parameters
        ----------
        """
        super(PuffinD, self).__init__()
        self.uplblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=17, padding=8),
                nn.BatchNorm1d(64)),

            nn.Sequential(
                nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
                nn.BatchNorm1d(96)),

            nn.Sequential(
                nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

        ])

        self.upblocks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(64, 64, fused=True),
                ConvBlock(64, 64, fused=True)),

            nn.Sequential(
                ConvBlock(96, 96, fused=True),
                ConvBlock(96, 96, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

        ])

        self.downlblocks = nn.ModuleList([

            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(128, 96, kernel_size=17, padding=8),
                nn.BatchNorm1d(96)),

            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(96, 64, kernel_size=17, padding=8),
                nn.BatchNorm1d(64)),

        ])

        self.downblocks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(96, 96, fused=True),
                ConvBlock(96, 96, fused=True)),

            nn.Sequential(
                ConvBlock(64, 64, fused=True),
                ConvBlock(64, 64, fused=True))

        ])

        self.uplblocks2 = nn.ModuleList([

            nn.Sequential(
                nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
                nn.BatchNorm1d(96)),

            nn.Sequential(
                nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

        ])

        self.upblocks2 = nn.ModuleList([

            nn.Sequential(
                ConvBlock(96, 96, fused=True),
                ConvBlock(96, 96, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

        ])

        self.downlblocks2 = nn.ModuleList([

            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(128, 96, kernel_size=17, padding=8),
                nn.BatchNorm1d(96)),

            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(96, 64, kernel_size=17, padding=8),
                nn.BatchNorm1d(64)),

        ])

        self.downblocks2 = nn.ModuleList([
            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
                ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(96, 96, fused=True),
                ConvBlock(96, 96, fused=True)),

            nn.Sequential(
                ConvBlock(64, 64, fused=True),
                ConvBlock(64, 64, fused=True))

        ])
        self.final = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 10, kernel_size=1),
            nn.Softplus())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = x
        encodings = []
        for i, lconv, conv in zip(np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        encodings2 = [out]
        for enc, lconv, conv in zip(reversed(encodings[:-1]), self.downlblocks, self.downblocks):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings2.append(out)

        encodings3 = [out]
        for enc, lconv, conv in zip(reversed(encodings2[:-1]), self.uplblocks2, self.upblocks2):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings3.append(out)

        for enc, lconv, conv in zip(reversed(encodings3[:-1]), self.downlblocks2, self.downblocks2):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out = self.final(out)
        return out


def PseudoPoissonKL(lpred, ltarget):
    return (ltarget * torch.log((ltarget + 1e-10) / (lpred + 1e-10)) + lpred - ltarget)


def figshow(x, np=False):
    if np:
        plt.imshow(x.squeeze())
    else:
        plt.imshow(x.squeeze().cpu().detach().numpy())
    plt.show()


def convert_pred_back_to_count_scale(y_hat):
    # > In this work, we refer to this aggregated signal as the log-scale signal
    # > and its inverse transformed value by 10^x − 1 as the count-scale signal.
    return np.power(10, y_hat) - 1.


def export_predictions_on_test(model, genome, chromosomes, chromosome_sizes, labels):
    assert len(labels) % 2 == 0
    n_cell_types = len(labels) // 2

    for chrom in chromosomes:
        bw_objs = [pyBigWig.open(f"{l}.{chrom}.bw", "w") for l in labels]
        bw_header = [(k, v) for k, v in chromosome_sizes.items()]
        for bo in bw_objs: bo.addHeader(bw_header)

        chr_len = chromosome_sizes[chrom]
        seq = genome.get_encoding_from_coords(chrom, 0, chr_len)
        chrom_placeholder = [chrom, ] * 50000
        with torch.no_grad():
            for ii in np.arange(0, chr_len, 50000)[:-2]:
                pred = model(
                    torch.FloatTensor(seq[ii: ii + 100000, :][None, :, :]).transpose(1, 2).cuda()
                ).cpu().detach().numpy()

                pred = convert_pred_back_to_count_scale(pred[0])
                pred[-n_cell_types:-1, :] = pred[-n_cell_types:-1, :] * -1

                starts = np.arange(ii + 25000, ii + 75000)
                ends = starts + 1

                for label_idx, target_pred in enumerate(pred):
                    bw_objs[label_idx].addEntries(chrom_placeholder, starts.tolist(), ends=ends.tolist(),
                                                  values=target_pred[25000:75000].tolist())
        for bo in bw_objs: bo.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pl-input", required=True,
                        type=str, nargs="+", help="Paths to the bigWig datasets.")
    parser.add_argument("-m", "--mn-input", required=True,
                        type=str, nargs="+", help="Paths to the bigWig datasets.")
    parser.add_argument("-l", "--label", required=True, type=str, nargs="+",
                        help="The non-redundant list of genomic features (i.e. labels) that will be predicted.")
    parser.add_argument("-f", "--fasta", required=True, type=str,
                        help="Path to the genome fasta file.")
    parser.add_argument("-t", "--test", required=True, type=str, nargs="+", )
    parser.add_argument("-v", "--validation", required=True, type=str, )
    parser.add_argument("-s", "--seed", default=3, type=int, help="Seed for sampling regions.")
    parser.add_argument("-o", "--model-str", default="puffin_D", type=str, help="Model name")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--max-epochs", default=500, type=int, help="Max epochs")
    parser.add_argument("-e", "--early-stop-epochs", default=3, type=int, help="Early stop epochs")
    parser.add_argument("-d", "--min-delta", default=0.0001, type=float,
                        help="Minimum incremental in the valid_cor to qualify as an improvement")

    args = parser.parse_args()

    if len(args.pl_input) != len(args.label) or len(args.pl_input) != len(args.mn_input):
        raise parser.error("The number of inputs must match the number of labels.")

    args.input = list(args.pl_input) + list(args.mn_input)
    args.comp_label = [f"{l}.pl" for l in args.label] + [f"{l}.mn" for l in args.label]

    with pyBigWig.open(args.pl_input[0]) as bw:
        args.chr_sizes = bw.chroms()

    return args


def main():
    args = get_args()
    seed = args.seed
    modelstr = args.model_str

    os.makedirs("./models/", exist_ok=True)

    tfeature = GenomicSignalFeatures(args.input, args.comp_label, (100000,), None)

    weights = torch.ones(len(args.input)).cuda()

    genome = selene_sdk.sequences.Genome(
        input_path=args.fasta,
        blacklist_regions="hg38"
    )

    noblacklist_genome = selene_sdk.sequences.Genome(
        input_path=args.fasta)

    sampler = RandomPositionsSampler(
        reference_sequence=genome,
        target=tfeature,
        features=[''],
        test_holdout=args.test,
        validation_holdout=[args.validation, ],
        sequence_length=100000,
        center_bin_to_predict=100000,
        position_resolution=1,
        random_shift=0,
        random_strand=False
    )

    sampler.mode = "train"
    dataloader = SamplerDataLoader(sampler, num_workers=16, batch_size=args.batch_size, seed=seed)

    try:
        net = torch.load('./models/' + modelstr + '.checkpoint')
    except:
        print("pretrained model not found")
        net = nn.DataParallel(PuffinD())
    print(f"Available GPUs: {torch.cuda.device_count()}")
    net.cuda()
    net.train()

    params = [p for p in net.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=0.005)
    try:
        temp = torch.load('./models/' + modelstr + '.optimizer')
        optimizer.load_state_dict(temp.state_dict())
    except:
        print("pretrained optimizer not found")

    i = 0
    past_losses = []
    firstvalid = True
    bestcor = 0
    epoch = 0
    time_to_early_stop = False
    epoch_of_best_loss = None
    while True:
        for sequence, target in dataloader:
            if torch.rand(1) < 0.5:
                sequence = sequence.flip([1, 2])
                target = target.flip([1, 2])

            optimizer.zero_grad()
            pred = net(torch.Tensor(sequence.float()).transpose(1, 2).cuda())
            loss = (PseudoPoissonKL(pred, target.cuda()) * weights[None, :, None]).mean()
            loss.backward()
            past_losses.append(loss.detach().cpu().numpy())

            optimizer.step()

            del pred
            del loss

            if i % 500 == 0:
                print(f"train loss ({i}): {np.mean(past_losses[-500:])}", flush=True)

            if i % 500 == 0:
                torch.save(net, './models/' + modelstr + '.checkpoint')
                torch.save(optimizer, './models/' + modelstr + '.optimizer')

            rstate_saved = np.random.get_state()
            if i % 8000 == 0:  # consider per 8000 steps as an epoch
                if firstvalid:
                    validseq = noblacklist_genome.get_encoding_from_coords(args.validation, 0,
                                                                           args.chr_sizes[args.validation])
                    valid_ref = tfeature.get_feature_data(args.validation, 0, args.chr_sizes[args.validation])
                    firstvalid = False
                net.eval()
                with torch.no_grad():
                    validpred = np.zeros((len(args.comp_label), args.chr_sizes[args.validation]))
                    for ii in np.arange(0, args.chr_sizes[args.validation], 50000)[:-2]:
                        pred = (
                            net(
                                torch.FloatTensor(validseq[ii: ii + 100000, :][None, :, :])
                                .transpose(1, 2).cuda()
                            ).cpu().detach().numpy()
                        )
                        pred2 = (
                            net(
                                torch.FloatTensor(validseq[ii: ii + 100000, :][None, ::-1, ::-1].copy())
                                .transpose(1, 2).cuda()
                            ).cpu().detach().numpy()[:, ::-1, ::-1]
                        )

                        validpred[:, ii + 25000: ii + 75000] = (
                                pred[0, :, 25000:75000] * 0.5 + pred2[0, :, 25000:75000] * 0.5
                        )

                valid_corrs = np.zeros(len(args.pl_input))
                for _idx in range(len(args.pl_input)):
                    valid_corrs[_idx] = (
                            np.corrcoef(validpred[_idx, :ii], valid_ref[_idx, :ii])[0, 1] * 0.5
                            + np.corrcoef(validpred[-(_idx + 1), :ii], valid_ref[-(_idx + 1), :ii])[0, 1] * 0.5
                    )
                valid_cor = np.sum(valid_corrs)
                print(f"Cor at step {i}, epoch {epoch}: {valid_corrs} ({valid_cor})")
                net.train()
                epoch += 1
                if epoch > args.max_epochs:
                    time_to_early_stop = True
                    break
                if bestcor < valid_cor - args.min_delta:
                    bestcor = valid_cor
                    epoch_of_best_loss = epoch
                    print(f"Saving optimal model at step {i}, epoch {epoch - 1}")
                    torch.save(net, './models/' + modelstr + '.best.checkpoint')
                    torch.save(optimizer, './models/' + modelstr + '.best.optimizer')
                else:
                    if epoch_of_best_loss is not None and epoch_of_best_loss <= epoch - args.early_stop_epochs:
                        time_to_early_stop = True
                        break

            i += 1
            if time_to_early_stop:
                break
        if time_to_early_stop:
            break

    net = torch.load('./models/' + modelstr + '.checkpoint')
    net.eval()
    print("Export results...")
    export_predictions_on_test(net, noblacklist_genome, args.test, args.chr_sizes, args.comp_label)


if __name__ == "__main__":
    main()
