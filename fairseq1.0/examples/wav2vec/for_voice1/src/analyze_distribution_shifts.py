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

def load_models(
    filenames, data_path, arg_overrides=None, task=None, model_state=None
):
    models = []
    criterions = []

    if arg_overrides is None:
        arg_overrides = {}

    arg_overrides["wer_args"] = None
    arg_overrides["data"] = data_path

    if filenames is None:
        assert model_state is not None
        filenames = [0]
    else:
        filenames = filenames.split(":")

    for filename in filenames:
        if model_state is None:
            if not os.path.exists(filename):
                raise IOError("Model file not found: {}".format(filename))
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
        model.load_state_dict(state["model"], strict=False)
        models.append(model)

    return models, task

def main(args, task=None, model_state=None):

    print("loading fine-tuned model(s) from {}".format(args.path))
    models, task = load_models(
        args.path,
        data_path=args.data,
        arg_overrides=eval(args.model_overrides),  # noqa
        task=task,
        model_state=model_state,
    )
    print('task is', task)
    finetune_model = models[0].w2v_encoder.w2v_model
    see_weight_rate(finetune_model)
    exit()

    print("loading pretrained model(s) from {}".format(args.pretrain_path))
    models, task = load_models(
        args.pretrain_path,
        data_path=args.data,
        arg_overrides=eval(args.model_overrides),  # noqa
        task=task,
        model_state=model_state,
    )
    print('task is', task)
    pretrain_model = models[0] 

    pretrain_model_dict = {}
    for name, param in pretrain_model.named_parameters():
        pretrain_model_dict[name] = param.data

    shift_dict = {}
    for name, param in finetune_model.named_parameters():
        diff = param.data - pretrain_model_dict[name] 
        if len(diff.shape) == 1:
            dim = diff.shape[0]
        elif len(diff.shape) == 2:
            dim = diff.shape[0] * diff.shape[1]
        shift = torch.linalg.norm(diff) / dim 
        #shift = torch.linalg.norm(diff) 
        shift_dict[name] = shift.numpy()
    
    # in ascending order by shift amount 
    sorted_shift_dict = {k: v for k, v in sorted(shift_dict.items(), key=lambda item: item[1])} 

    for k, v in sorted_shift_dict.items():
        print(k, v)

    #print(model.feature_extractor)
    #print(model.post_extract_proj)
    
    ## none -- they are only used during CPC pretraining 
    #print(model.quantizer)
    #print(model.project_q)

    ## BERT 
    #print(model.encoder)
    #print(model.feature_extractor)

def see_weight_rate(model):

    total, target = 0, 0 
    for name, param in model.named_parameters():
        total += float(param.numel())
        if 'pos_conv' in name: 
            target += float(param.numel())

    print('target layers consists of {}% of original model'.format(target / total * 100))


def add_custom_argument(parser):
    parser.add_argument('--pretrain-path', type=str, help='pretrain wav2vec path')
    parser.add_argument('--outdir', type=str, help='output directory')

    return parser 

def make_parser():
    parser = options.get_generation_parser()
    parser = add_custom_argument(parser)
    
    return parser


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
