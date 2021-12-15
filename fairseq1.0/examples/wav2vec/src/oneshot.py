#!/usr/bin/env python3 -u

import logging
import math
import os
import sys

import editdistance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging.meters import StopwatchMeter, TimeMeter

from pruning_methods import *

def load_models(
    filenames, data_path, arg_overrides=None, task=None, model_state=None, load_state=True
):
    models = []
    criterions = []

    if arg_overrides is None:
        arg_overrides = {}

    arg_overrides["wer_args"] = None
    arg_overrides["data"] = data_path

    if filenames is None:
        #assert model_state is not None
        filenames = [0]
    else:
        filenames = filenames.split(":")

    for filename in filenames:
        if model_state is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)
        else:
            state = model_state

        if "cfg" in state:
            cfg = state["cfg"]
        else:
            cfg = convert_namespace_to_omegaconf(state["args"])

        if task is None:
            if hasattr(cfg.task, 'data'):
                cfg.task.data = data_path
            task = tasks.setup_task(cfg.task)

        model = task.build_model(cfg.model)
        
        if load_state:
            model.load_state_dict(state["model"], strict=True)
        else:
            print('model state not loaded!')
        models.append(model)

    return models, task

def main(args, task=None, model_state=None):
    print("loading model(s) from {}".format(args.path))
    models, task = load_models(
        args.path,
        data_path=args.data,
        arg_overrides=eval(args.model_overrides),  # noqa
        task=task,
        model_state=model_state,
        load_state=not args.random_init
    )

    # add fp16
    for model in models:
        if args.fp16:
            print('model on fp16')
            model.half()
        #model.cuda()

    if args.omp_cpc:
        # for pretrained wav2vec 
        print('model is a pretrained wav2vec')
        model = models[0]
    else:
        # for fine-tuned wav2vec 
        print('model is a finetuned wav2vec')
        model = models[0].w2v_encoder.w2v_model

    ## feature extractor -- present in pretrained and fine-tuned model 
    #print(model.feature_extractor)
    #print(model.post_extract_proj)

    ## quantizer -- only present in pretrained model 
    #print(model.quantizer)
    #print(model.project_q)

    ## BERT -- present in pretrained and fine-tuned model
    #print(model.encoder)

    with torch.no_grad():
        if args.prune_component == 'feature_extractor':
            pruning_feature_extractor(model, args.rate)
        elif args.prune_component == 'quantizer':
            pruning_quantizer(model, args.rate)
        elif args.prune_component == 'bert':
            pruning_bert(model, args.rate, args.model_type)
        elif args.prune_component == 'all':
            pruning_all(model, args.rate)
        else:
            print('pruning %s not supported' % args.prune_component)
            exit()

        feature_extractor_zero_rate, quantizer_zero_rate, bert_zero_rate, total_zero_rate = \
            see_weight_rate(model, args.model_type)
        print('wav2vec 2.0 feature extractor zero rate %f' % feature_extractor_zero_rate)
        if args.omp_cpc:
            print('wav2vec 2.0 quantizer zero rate %f' % quantizer_zero_rate)
        print('wav2vec 2.0 BERT zero rate %f' % bert_zero_rate)
        print('wav2vec 2.0 total_zero_rate %f' % total_zero_rate)

        mask_dict = {}
        weight_dict = {}
        model_dict = model.state_dict()
        for key in model_dict.keys():
            if 'mask' in key:
                mask_dict[key] = model_dict[key]
            else:
                weight_dict[key] = model_dict[key]

        torch.save(mask_dict, os.path.join(args.outdir, args.prune_component + '_' + str(args.rate) + '_mask.pt'))
        torch.save(weight_dict, os.path.join(args.outdir, args.prune_component + '_' + str(args.rate) + '_weight.pt'))

def add_argument(parser):
    parser.add_argument('--model-type', default='wav2vec_small', choices=['wav2vec_small', 'libri960_big'],
                        type=str, help='pretrained wav2vec model architecture (BASE or LARGE)')
    parser.add_argument('--random-init', action='store_true', help='random initialize wav2vec')
    parser.add_argument('--omp-cpc', action='store_true', help='pruning pretrained wav2vec 2.0')

    return parser 

def make_parser():
    parser = options.get_generation_parser()
    parser = add_pruning_argument(parser)
    parser = add_argument(parser)

    return parser


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
