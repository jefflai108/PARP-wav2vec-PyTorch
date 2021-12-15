# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import math
import sys
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, tasks, utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer

from .wav2vec2_asr import Linear, Embedding, TransformerModel, TransformerDecoder, add_common_args

sys.path.append('/nobackup/users/clai24/lth-wav2vec/lth-wav2vec/examples/wav2vec/src')
from pruning_methods import *

@register_model("wav2vec_ctc_oneshot_lid")
class Wav2VecCtcOneShot(BaseFairseqModel):
    """ ++ pruning option to wav2vec2_asr.Wav2VecCtc
    """
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        add_common_args(parser)

    def __init__(self, w2v_encoder, args):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.args = args

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, args, task, pruning=False):
        base_architecture(args)
        w2v_encoder = Wav2VecEncoderOneShot(args, task.target_dictionary, args.mask_file)
        return cls(w2v_encoder, args)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x

    # def max_positions(self):
    #     return None


class Wav2VecEncoderOneShot(FairseqEncoder):
    """ ++ pruning option to wav2vec2_asr.Wav2VecEncoder
    """
    def __init__(self, args, tgt_dict=None, pruning_mask=None):
        """ 1. build model
            2. load pretrained wav2vec 2.0 weights
            3. apply pruning mask
        """
        self.apply_mask = args.apply_mask

        arg_overrides = {
            "dropout": args.dropout,
            "activation_dropout": args.activation_dropout,
            "dropout_input": args.dropout_input,
            "attention_dropout": args.attention_dropout,
            "mask_length": args.mask_length,
            "mask_prob": args.mask_prob,
            "mask_selection": args.mask_selection,
            "mask_other": args.mask_other,
            "no_mask_overlap": args.no_mask_overlap,
            "mask_channel_length": args.mask_channel_length,
            "mask_channel_prob": args.mask_channel_prob,
            "mask_channel_selection": args.mask_channel_selection,
            "mask_channel_other": args.mask_channel_other,
            "no_mask_channel_overlap": args.no_mask_channel_overlap,
            "encoder_layerdrop": args.layerdrop,
            "feature_grad_mult": args.feature_grad_mult,
        }

        if getattr(args, "w2v_args", None) is None:
            print('loading pretrained wav-file from {}'.format(args.w2v_path))
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.w2v_path, arg_overrides
            )
            w2v_args = state.get("args", None) or state["cfg"].model
        else:
            state = None
            w2v_args = args.w2v_args

        assert (
            args.normalize == w2v_args.normalize
        ), "Fine-tuning works best when data normalization is the same"

        w2v_args.data = args.data
        task = tasks.setup_task(w2v_args)
        model = task.build_model(w2v_args)

        if state is not None and not args.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        # this removes the quantizer as it is not used during fine-tuning
        model.remove_pretraining_modules()

        # #################################################################
        # masking code snippet below is moved to trainer class at
        # /data/sls/temp/clai24/lottery-ticket/fairseq/fairseq/trainer.py
        # #################################################################

        #feature_extractor_zero_rate, bert_zero_rate, total_zero_rate = see_weight_rate(model)
        #print('Before pruning: wav2vec 2.0 feature extractor zero rate %f' % feature_extractor_zero_rate)
        #print('Before pruning: wav2vec 2.0 BERT zero rate %f' % bert_zero_rate)
        #print('Before pruning: wav2vec 2.0 total_zero_rate %f' % total_zero_rate)

        ## apply masking
        ## hack: move both model and mask to cuda and apply this operation
        ## this only works on 1 GPU
        #if pruning_mask is not None:
        #    model.cuda()
        #    mask = torch.load(pruning_mask, map_location='cuda:0')
        #    #device = model.mask_emb.device
        #    #mask = torch.load(pruning_mask, map_location=device)
        #    apply_pruning_mask(model, mask)
        #    feature_extractor_zero_rate, bert_zero_rate, total_zero_rate = see_weight_rate(model)
        #    print('After pruning: wav2vec 2.0 feature extractor zero rate %f' % feature_extractor_zero_rate)
        #    print('After pruning: wav2vec 2.0 BERT zero rate %f' % bert_zero_rate)
        #    print('After pruning: wav2vec 2.0 total_zero_rate %f' % total_zero_rate)

        super().__init__(task.source_dictionary)

        d = w2v_args.encoder_embed_dim

        # assign model here
        self.w2v_model = model

        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(args, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, args.decoder_embed_dim)
        else:
            self.proj = None

        import pickle
        with open("/nobackup/users/clai24/lth-wav2vec/lth-wav2vec/examples/wav2vec/test_lan_embed.pickle", "rb") as f:
            self.lan_embed = pickle.load(f)['train']['es'].cuda()
        self.lan_embed_proj = Linear(576, 100)
        self.combin_lan_and_bert = Linear(len(tgt_dict)*2, len(tgt_dict))
        self.lan_dropout = nn.Dropout(args.final_dropout)
        self.proj = Linear(d+100, len(tgt_dict))
        print('mother-fucker')
        exit()

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        # ft means not updating wav2vec 2.0
        # and only updates the projection layer until
        # self.freeze_finetune_updates is reached
        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)
        #print(x.shape) # torch.Size([B, T, 768])
        #print(self.lan_embed.shape) # torch.Size([576])
        lan_embed = self.lan_embed_proj(self.lan_embed).repeat(x.shape[0], x.shape[1], 1)
        lan_embed = self.lan_dropout(F.tanh(lan_embed))
        #print(lan_embed.shape) # torch.Size([B, T, 100+768])
        #x = self.combin_lan_and_bert(F.relu(torch.cat((x, lan_embed), dim=-1)))
        x = torch.cat((x, lan_embed), dim=-1)
        if self.proj:
            x = self.proj(x)
            #print(self.proj.weight.shape)
            #print('proj layer: ssl vs lan weights', self.proj.weight[:768,:].data.abs().mean(), self.proj.weight[768:,:].data.abs().mean())

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

@register_model_architecture("wav2vec_ctc_oneshot_lid", "wav2vec_ctc_oneshot_lid")
def base_architecture(args):
    args.no_pretrained_weights = getattr(args, "no_pretrained_weights", False)
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.final_dropout = getattr(args, "final_dropout", 0)
    args.apply_mask = getattr(args, "apply_mask", False)
    args.dropout = getattr(args, "dropout", 0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)

    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0)
    args.layerdrop = getattr(args, "layerdrop", 0.0)

