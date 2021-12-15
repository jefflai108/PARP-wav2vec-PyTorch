#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import logging
import math
import os
import sys
from tqdm import tqdm
import shutil

import editdistance
import numpy as np
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging.meters import StopwatchMeter, TimeMeter
import torch.optim as optim

# custom below
import soundfile as sf
from transformers import Wav2Vec2Processor
from dataclass import LIDDataset, make_pad_mask
from lid_models import LID_XLSR53
from optimizer import ScheduledOptim

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_asr_eval_argument(parser):
    parser.add_argument(
        "--inter-lan-entropy",
        action="store_true",
        help="inter-lan entropy over codebooks",
    )
    parser.add_argument(
        "--straight-through",
        action="store_true",
        help="set attention scores with >0.5 to 1 and <0.5 to 0 during forward pass",
    )
    parser.add_argument(
        "--lan-residual",
        action="store_true",
        help="lan_emb residual connection before classification",
    )
    parser.add_argument(
        "--use-spec",
        action="store_true",
        help="spectrogram instead of waveform as input features",
    )
    parser.add_argument(
        "--trun-on-diversity-loss",
        action="store_true",
        help="enable diversity loss",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="debug mode. everything's faster",
    )
    parser.add_argument(
        "--xlsr-pretrained",
        action="store_true",
        help="load pretrained xlsr",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="log directory",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=100000,
        help="seq len for training models",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="train/eval mode",
    )
    parser.add_argument(
        "--lan-codebooks",
        type=int,
        default=4,
        help="num of lan templates",
    )
    parser.add_argument(
        "--lan-codebook-dim",
        type=int,
        default=128,
        help="codebook embed dimension",
    )
    parser.add_argument(
        "--num-epoches",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--diversity-alpha",
        type=float,
        default=0.1,
        help="diversity loss weighing term"
    )
    parser.add_argument(
        "--batch-num",
        type=int,
        default=256,
    )

    return parser

def main(args, task=None, model_state=None):

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")

    lid_model = LID_XLSR53(args, logger).to(device)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    if args.debug_mode:
        train_class = LIDDataset(args, 'data/commonvoice-all-train/all-debug.tsv', processor, args.seq_len)
        val_class   = LIDDataset(args, 'data/commonvoice-all-train/all-debug.tsv', processor)
        test_class  = LIDDataset(args, 'data/commonvoice-all-train/all-debug.tsv', processor)
    else:
        train_class = LIDDataset(args, 'data/commonvoice-all-train/all-train.tsv', processor, args.seq_len)
        val_class   = LIDDataset(args, 'data/commonvoice-all-train/all-val.tsv', processor)
        test_class  = LIDDataset(args, 'data/commonvoice-all-train/all-test.tsv', processor)

    kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_class, batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_class, batch_size=1, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_class, batch_size=1, shuffle=False, **kwargs)
    model_params = sum(p.numel() for p in lid_model.parameters() if p.requires_grad)
    logger.info('===> Model total parameter: {}'.format(model_params))

    criterion = torch.nn.NLLLoss()

    all_avg_lan_embed = {}
    logger.info("evaluating best model on val and test")
    checkpoint = torch.load(os.path.join(args.log_dir, 'model_best.pth.tar'), lambda a,b:a)
    lid_model.load_state_dict(checkpoint["state_dict"])
    logger.info('testing on train set')
    train_avg_lan_embed = val_and_eval_step(args, logger, lid_model, train_loader, criterion, device, mode='Train')
    all_avg_lan_embed['train'] = train_avg_lan_embed
    logger.info('testing on val set')
    val_avg_lan_embed = val_and_eval_step(args, logger, lid_model, val_loader, criterion, device, mode='Val')
    all_avg_lan_embed['val'] = all_avg_lan_embed
    logger.info('testing on Test set')
    test_avg_lan_embed = val_and_eval_step(args, logger, lid_model, test_loader, criterion, device, mode='Test')
    all_avg_lan_embed['test'] = test_avg_lan_embed

    import pickle
    with open('test_lan_embed.pickle', 'wb') as handle:
        pickle.dump(all_avg_lan_embed, handle)

    #flac_paths = ['common_voice_es_18344695.flac', 'common_voice_es_18859741.flac', 'common_voice_es_18841133.flac', 'common_voice_es_18883207.flac', 'common_voice_es_18336453.flac']
    #for flac_path in flac_paths:
    #    input = process_audio(lid_model.xlsr53, processor, os.path.join('/home/clai24/common_voices_splits/es/clips_16k-test', flac_path))
    #    lid_model(input)

def val_and_eval_step(args, logger, model, loader, criterion, device, mode='val'):
    model.eval()
    test_loss = 0
    correct = 0
    diversity_scores = 0
    lans = ['es', 'fr', 'it', 'ky', 'nl', 'ru', 'sv_SE', 'tr', 'tt', 'zh_TW']
    lan2attention_score = {x:torch.zeros(args.lan_codebook_dim*(args.lan_codebooks+1)) for ii, x in enumerate(lans)}
    lan2cnt = {x:0 for ii, x in enumerate(lans)}
    with torch.no_grad():
        for inputs, targets, true_len in tqdm(loader):
            inputs   = inputs.to(device)
            targets  = targets.to(device)
            true_len = true_len.to(device)
            padding_mask = make_pad_mask(true_len)
            outputs, _, diversity_score = model(inputs, padding_mask)
            lan_embed = model.extract_lan_embed(inputs, padding_mask)
            #print(lan_embed.shape)
            diversity_scores += diversity_score.item() * 1 # batch size 1
            if not args.trun_on_diversity_loss: # set diversity loss to 0
                diversity_score = 0; diversity_scores = 0
            test_loss += criterion(outputs, targets).item() - args.diversity_alpha * diversity_score # sum up batch loss
            pred = outputs.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
            target_lan = lans[targets.item()]
            lan2attention_score[target_lan] += torch.sum(lan_embed, dim=0).squeeze().cpu()
            lan2cnt[target_lan] += 1
    mean_diversity_score = diversity_scores / len(loader.dataset)
    # extract_lan_embed
    test_loss /= len(loader.dataset)
    logger.info('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\tDiversity: {:.4f}\n'.format(
        mode, test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset), mean_diversity_score
    ))
    logger.info('\nsampled lan_embed distribution is:')
    avg_lan_embed = {}
    for lan, lan_embed in lan2attention_score.items():
        avg_lan_embed[lan] = lan_embed / lan2cnt[lan]

    return avg_lan_embed

def process_audio(model, processor, flac_path):
    """ convert raw flac file to wav2vec2 acceptable input leveraging Huggingface's processor
    """
    wav, samplerate = sf.read(flac_path)
    input_tensors = processor(wav, return_tensorzs="pt", sampling_rate=samplerate).input_values
    input_tensors = torch.tensor(input_tensors).float().cuda()

    return input_tensors


def make_parser():
    parser = options.get_generation_parser()
    parser = add_asr_eval_argument(parser)
    return parser

def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
