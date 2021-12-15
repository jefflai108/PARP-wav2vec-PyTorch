import os
import sys

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def rewind(pre_weight, model_type='wav2vec_small'):
    if model_type == 'wav2vec_small':
        num_transformer_blocks = 12
    elif model_type == 'libri960_big' or model_type == 'xlsr_53_56k':
        num_transformer_blocks = 24
    else:
        print('model type {} not supported'.format(model_type))
        exit()
    print('num_transformer_blocks is', num_transformer_blocks)

    recover_dict = {}
    name_list = []
    for ii in range(num_transformer_blocks):
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.self_attn.k_proj.weight')
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.self_attn.k_proj.bias')
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.self_attn.v_proj.weight')
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.self_attn.v_proj.bias')
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.self_attn.q_proj.weight')
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.self_attn.q_proj.bias')
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.self_attn.out_proj.weight')
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.self_attn.out_proj.bias')
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.fc1.weight')
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.fc1.bias')
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.fc2.weight')
        name_list.append('w2v_encoder.w2v_model.encoder.layers.'+str(ii)+'.fc2.bias')

    for key in pre_weight.keys():
        if 'w2v_encoder' in key:
            if key in name_list:
                new_key = key+'_orig'
                #print('updating to', new_key)
            else:
                new_key = key

            recover_dict[new_key] = pre_weight[key]

    return recover_dict

def add_pruning_argument(parser):
    parser.add_argument('--prune-component', default='bert', choices=['bert', 'feature_extractor', 'quantizer', 'all'],
                        type=str, help='prune on bert or feature_extractor')
    parser.add_argument('--rate', default=0.2, type=float, help='rate')
    parser.add_argument('--outdir', type=str, help='output directory for storing mask and weight.pt')

    return parser

def pruning_feature_extractor(model, px):
    """
    Note: Since feature extractor is not updated during fine-tuning, we
          decide not to prune it.

    prune out wav2vec 2.0 feature extractor. Its bias was set to false
    so only weights are pruned.
    """
    parameters_to_prune =[]
    for ii in range(7):
        parameters_to_prune.append((model.feature_extractor.conv_layers[ii][0], 'weight'))

    parameters_to_prune.append((model.post_extract_proj, 'weight'))
    parameters_to_prune.append((model.post_extract_proj, 'bias'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def pruning_quantizer(model, px):
    """
    prune out wav2vec 2.0 quantizer.
    """
    parameters_to_prune =[]
    parameters_to_prune.append((model.quantizer.weight_proj, 'weight'))
    parameters_to_prune.append((model.quantizer.weight_proj, 'bias'))

    parameters_to_prune.append((model.project_q, 'weight'))
    parameters_to_prune.append((model.project_q, 'bias'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def unprune_bert(model, model_type='wav2vec_small'):
    """
    remove pruning forward pre-hook. This is useful when we want to tweek the learned pruned mask.
    """
    if model_type == 'wav2vec_small':
        num_transformer_blocks = 12
    elif model_type == 'libri960_big' or model_type == 'xlsr_53_56k':
        num_transformer_blocks = 24
    else:
        print('model type {} not supported'.format(model_type))
        exit()
    print('num_transformer_blocks is', num_transformer_blocks)

    parameters_to_prune =[]
    # position encoding for BERT
    #model.encoder.pos_conv[0].weight = nn.Parameter(model.encoder.pos_conv[0].weight,
    #                                                requires_grad=True) # hack to make it work
    #model.encoder.pos_conv[0].bias = nn.Parameter(model.encoder.pos_conv[0].bias,
    #                                                requires_grad=True) # hack
    #parameters_to_prune.append((model.encoder.pos_conv[0], 'weight'))
    #parameters_to_prune.append((model.encoder.pos_conv[0], 'bias'))

    for ii in range(num_transformer_blocks):
        parameters_to_prune.append(model.encoder.layers[ii].self_attn.k_proj)
        parameters_to_prune.append(model.encoder.layers[ii].self_attn.v_proj)
        parameters_to_prune.append(model.encoder.layers[ii].self_attn.q_proj)
        parameters_to_prune.append(model.encoder.layers[ii].self_attn.out_proj)
        parameters_to_prune.append(model.encoder.layers[ii].fc1)
        parameters_to_prune.append(model.encoder.layers[ii].fc2)

    for ii in range(0, len(parameters_to_prune)): # applying both weight+bias masks
        prune.remove(parameters_to_prune[ii], 'weight')
        prune.remove(parameters_to_prune[ii], 'bias')

def pruning_bert(model, px, model_type='wav2vec_small'):
    """
    prune out wav2vec 2.0 BERT: 12 transformer layers for BASE, and 24
                                transformer layers for LARGE

    note: position encoding and projection heads are not pruned.
    """
    if model_type == 'wav2vec_small':
        num_transformer_blocks = 12
    elif model_type == 'libri960_big' or model_type == 'xlsr_53_56k':
        num_transformer_blocks = 24
    else:
        print('model type {} not supported'.format(model_type))
        exit()
    print('num_transformer_blocks is', num_transformer_blocks)

    parameters_to_prune =[]
    # position encoding for BERT
    #model.encoder.pos_conv[0].weight = nn.Parameter(model.encoder.pos_conv[0].weight,
    #                                                requires_grad=True) # hack to make it work
    #model.encoder.pos_conv[0].bias = nn.Parameter(model.encoder.pos_conv[0].bias,
    #                                                requires_grad=True) # hack
    #parameters_to_prune.append((model.encoder.pos_conv[0], 'weight'))
    #parameters_to_prune.append((model.encoder.pos_conv[0], 'bias'))

    for ii in range(num_transformer_blocks):
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.k_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.k_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.v_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.v_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.q_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.q_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.out_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.out_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].fc1, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].fc1, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].fc2, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].fc2, 'bias'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def pruning_all(model, px):
    """
    prune out wav2vec 2.0 (CNN+BERT).
    """
    parameters_to_prune =[]

    # feature_extractor
    for ii in range(7):
        parameters_to_prune.append((model.feature_extractor.conv_layers[ii][0], 'weight'))

    parameters_to_prune.append((model.post_extract_proj, 'weight'))
    parameters_to_prune.append((model.post_extract_proj, 'bias'))

    # BERT
    #model.encoder.pos_conv[0].weight = nn.Parameter(model.encoder.pos_conv[0].weight,
    #                                                requires_grad=True) # hack to make it work
    #model.encoder.pos_conv[0].bias = nn.Parameter(model.encoder.pos_conv[0].bias,
    #                                                requires_grad=True) # hack
    #parameters_to_prune.append((model.encoder.pos_conv[0], 'weight'))
    #parameters_to_prune.append((model.encoder.pos_conv[0], 'bias'))

    for ii in range(12):
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.k_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.k_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.v_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.v_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.q_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.q_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.out_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.out_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].fc1, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].fc1, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].fc2, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].fc2, 'bias'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def see_weight_rate(model, model_type='wav2vec_small'):

    if model_type == 'wav2vec_small':
        num_transformer_blocks = 12
    elif model_type == 'libri960_big' or model_type == 'xlsr_53_56k':
        num_transformer_blocks = 24
    else:
        print('model type {} not supported'.format(model_type))
        exit()

    # feature extractor
    sum_list_1, zero_sum_1 = 0, 0
    for ii in range(7):
        sum_list_1 = sum_list_1 + float(model.feature_extractor.conv_layers[ii][0].weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.feature_extractor.conv_layers[ii][0].weight == 0))

    sum_list_1 = sum_list_1 + float(model.post_extract_proj.weight.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.post_extract_proj.weight == 0))
    sum_list_1 = sum_list_1 + float(model.post_extract_proj.bias.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.post_extract_proj.bias == 0))

    # quantizer
    sum_list_3, zero_sum_3 = 0, 0
    if model.quantizer is not None:
        sum_list_3 = sum_list_3 + float(model.quantizer.weight_proj.weight.nelement())
        zero_sum_3 = zero_sum_3 + float(torch.sum(model.quantizer.weight_proj.weight == 0))
        sum_list_3 = sum_list_3 + float(model.quantizer.weight_proj.bias.nelement())
        zero_sum_3 = zero_sum_3 + float(torch.sum(model.quantizer.weight_proj.bias == 0))

        sum_list_3 = sum_list_3 + float(model.project_q.weight.nelement())
        zero_sum_3 = zero_sum_3 + float(torch.sum(model.project_q.weight == 0))
        sum_list_3 = sum_list_3 + float(model.project_q.bias.nelement())
        zero_sum_3 = zero_sum_3 + float(torch.sum(model.project_q.bias == 0))

    # BERT
    sum_list_2, zero_sum_2 = 0, 0
    #sum_list_2 = sum_list_2 + float(model.encoder.pos_conv[0].weight.nelement())
    #zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.pos_conv[0].weight == 0))
    #sum_list_2 = sum_list_2 + float(model.encoder.pos_conv[0].bias.nelement())
    #zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.pos_conv[0].bias == 0))

    for ii in range(num_transformer_blocks):
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].self_attn.k_proj.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].self_attn.k_proj.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].self_attn.k_proj.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].self_attn.k_proj.bias == 0))

        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].self_attn.v_proj.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].self_attn.v_proj.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].self_attn.v_proj.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].self_attn.v_proj.bias == 0))

        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].self_attn.q_proj.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].self_attn.q_proj.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].self_attn.q_proj.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].self_attn.q_proj.bias == 0))

        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].self_attn.out_proj.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].self_attn.out_proj.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].self_attn.out_proj.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].self_attn.out_proj.bias == 0))

        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].fc1.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].fc1.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].fc1.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].fc1.bias == 0))

        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].fc2.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].fc2.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].fc2.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].fc2.bias == 0))

    feature_extractor_zero_rate = 100 * zero_sum_1 / sum_list_1
    if model.quantizer is not None:
        quantizer_zero_rate = 100 * zero_sum_3 / sum_list_3
    else:
        quantizer_zero_rate = None
    bert_zero_rate = 100 * zero_sum_2 / sum_list_2
    total_sum = (sum_list_1 + sum_list_2 + sum_list_3)
    total_zero_rate = 100 * (zero_sum_1 + zero_sum_2 + zero_sum_3) / total_sum
    print('feature_extractor:quantizer:BERT model param == {0:.2f}:{1:.2f}:{2:.2f}'.
            format(sum_list_1/total_sum, sum_list_3/total_sum, sum_list_2/total_sum))

    if quantizer_zero_rate is not None:
        return feature_extractor_zero_rate, quantizer_zero_rate, bert_zero_rate, total_zero_rate
    else:
        return feature_extractor_zero_rate, bert_zero_rate, total_zero_rate

def apply_pruning_mask(model, mask_dict, prune_component='bert', model_type='wav2vec_small', fp16=False):
    """
    apply pruning mask to wav2vec 2.0. We only prune either just the quantizer or just the BERT.
    """

    if prune_component not in ['bert', 'quantizer']:
        print('{} not supported'.format(prune_component))
        exit()

    if model_type == 'wav2vec_small':
        num_transformer_blocks = 12
    elif model_type == 'libri960_big' or model_type == 'xlsr_53_56k':
        num_transformer_blocks = 24
    else:
        print('model type {} not supported'.format(model_type))
        exit()

    #if fp16:
    #    model = model.half() # do this for fp16

    parameters_to_prune =[]
    mask_list_w, mask_list_b = [], [] # maks list for weight and bias

    if prune_component == 'feature_extractor':
        # feature_extractor
        for ii in range(7):
            parameters_to_prune.append(model.feature_extractor.conv_layers[ii][0])
            mask_list_w.append(mask_dict['feature_extractor.conv_layers.' + str(ii) + '.0.weight_mask'])

        parameters_to_prune.append(model.post_extract_proj)
        mask_list_w.append(mask_dict['post_extract_proj.weight_mask'])
        mask_list_b.append(mask_dict['post_extract_proj.bias_mask'])

    if prune_component == 'quantizer':
        parameters_to_prune.append(model.quantizer.weight_proj)
        mask_list_w.append(mask_dict['quantizer.weight_proj.weight_mask'])
        mask_list_w.append(mask_dict['quantizer.weight_proj.bias_mask'])

        parameters_to_prune.append(model.project_q)
        mask_list_w.append(mask_dict['project_q.weight_mask'])
        mask_list_w.append(mask_dict['project_q.bias_mask'])

    # BERT
    #model.encoder.pos_conv[0].weight = nn.Parameter(model.encoder.pos_conv[0].weight,
    #                                                requires_grad=True) # hack to make it work
    #model.encoder.pos_conv[0].bias = nn.Parameter(model.encoder.pos_conv[0].bias,
    #                                                requires_grad=True) # hack
    #parameters_to_prune.append(model.encoder.pos_conv[0])
    #mask_list_w.append(mask_dict['encoder.pos_conv.0.weight_mask'])
    #mask_list_b.append(mask_dict['encoder.pos_conv.0.bias_mask'])

    if prune_component == 'bert':
        for ii in range(num_transformer_blocks):
            parameters_to_prune.append(model.encoder.layers[ii].self_attn.k_proj)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.k_proj.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.k_proj.bias_mask'])
            parameters_to_prune.append(model.encoder.layers[ii].self_attn.v_proj)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.v_proj.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.v_proj.bias_mask'])
            parameters_to_prune.append(model.encoder.layers[ii].self_attn.q_proj)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.q_proj.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.q_proj.bias_mask'])
            parameters_to_prune.append(model.encoder.layers[ii].self_attn.out_proj)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.out_proj.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.out_proj.bias_mask'])
            parameters_to_prune.append(model.encoder.layers[ii].fc1)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.fc1.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.fc1.bias_mask'])
            parameters_to_prune.append(model.encoder.layers[ii].fc2)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.fc2.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.fc2.bias_mask'])

    #for ii in range(7): # only applying weight mask
    #    prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list_w[ii])
    #for ii in range(7, len(parameters_to_prune)): # applying both weight+bias masks
    #    prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list_w[ii])
    #    prune.CustomFromMask.apply(parameters_to_prune[ii], 'bias', mask=mask_list_b[ii-7])
    for ii in range(0, len(parameters_to_prune)): # applying both weight+bias masks
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list_w[ii])
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'bias', mask=mask_list_b[ii])

def apply_pruning_mask2(model, mask_dict):
    """
    apply pruning mask to wav2vec 2.0 (CNN+BERT)
    *without* in-place operation using
    https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.custom_from_mask.html
    """

    import copy
    #pruned_model = copy.deepcopy(model)
    parameters_to_prune =[]

    # feature_extractor
    for ii in range(7):
        #parameters_to_prune.append(model.feature_extractor.conv_layers[ii][0])
        model.feature_extractor.conv_layers[ii][0] = prune.custom_from_mask(
            model.feature_extractor.conv_layers[ii][0], name='weight', mask=
            mask_dict['feature_extractor.conv_layers.' + str(ii) + '.0.weight_mask']
        )

    #parameters_to_prune.append(model.post_extract_proj)
    model.post_extract_proj = prune.custom_from_mask(
        model.post_extract_proj, name='weight', mask=
        mask_dict['post_extract_proj.weight_mask']
    )
    model.post_extract_proj = prune.custom_from_mask(
        model.post_extract_proj, name='bias', mask=
        mask_dict['post_extract_proj.bias_mask']
    )

    # BERT
    #model.encoder.pos_conv[0].weight = nn.Parameter(model.encoder.pos_conv[0].weight,
    #                                                requires_grad=True) # hack to make it work
    #model.encoder.pos_conv[0].bias = nn.Parameter(model.encoder.pos_conv[0].bias,
    #                                                requires_grad=True) # hack
    #parameters_to_prune.append(model.encoder.pos_conv[0])
    #mask_list_w.append(mask_dict['encoder.pos_conv.0.weight_mask'])
    #mask_list_b.append(mask_dict['encoder.pos_conv.0.bias_mask'])

    for ii in range(12):
        #parameters_to_prune.append(model.encoder.layers[ii].self_attn.k_proj)
        model.encoder.layers[ii].self_attn.k_proj = prune.custom_from_mask(
            model.encoder.layers[ii].self_attn.k_proj, name='weight', mask=
            mask_dict['encoder.layers.' + str(ii) + '.self_attn.k_proj.weight_mask']
        )
        model.encoder.layers[ii].self_attn.k_proj = prune.custom_from_mask(
            model.encoder.layers[ii].self_attn.k_proj, name='bias', mask=
            mask_dict['encoder.layers.' + str(ii) + '.self_attn.k_proj.bias_mask']
        )

        #parameters_to_prune.append(model.encoder.layers[ii].self_attn.v_proj)
        model.encoder.layers[ii].self_attn.v_proj = prune.custom_from_mask(
            model.encoder.layers[ii].self_attn.v_proj, name='weight', mask=
            mask_dict['encoder.layers.' + str(ii) + '.self_attn.v_proj.weight_mask']
        )
        model.encoder.layers[ii].self_attn.v_proj = prune.custom_from_mask(
            model.encoder.layers[ii].self_attn.v_proj, name='bias', mask=
            mask_dict['encoder.layers.' + str(ii) + '.self_attn.v_proj.bias_mask']
        )

        #parameters_to_prune.append(model.encoder.layers[ii].self_attn.q_proj)
        model.encoder.layers[ii].self_attn.q_proj = prune.custom_from_mask(
            model.encoder.layers[ii].self_attn.q_proj, name='weight', mask=
            mask_dict['encoder.layers.' + str(ii) + '.self_attn.q_proj.weight_mask']
        )
        model.encoder.layers[ii].self_attn.q_proj = prune.custom_from_mask(
            model.encoder.layers[ii].self_attn.q_proj, name='bias', mask=
            mask_dict['encoder.layers.' + str(ii) + '.self_attn.q_proj.bias_mask']
        )

        #parameters_to_prune.append(model.encoder.layers[ii].self_attn.out_proj)
        model.encoder.layers[ii].self_attn.out_proj = prune.custom_from_mask(
            model.encoder.layers[ii].self_attn.out_proj, name='weight', mask=
            mask_dict['encoder.layers.' + str(ii) + '.self_attn.out_proj.weight_mask']
        )
        model.encoder.layers[ii].self_attn.out_proj = prune.custom_from_mask(
            model.encoder.layers[ii].self_attn.out_proj, name='bias', mask=
            mask_dict['encoder.layers.' + str(ii) + '.self_attn.out_proj.bias_mask']
        )

        #parameters_to_prune.append(model.encoder.layers[ii].fc1)
        model.encoder.layers[ii].fc1 = prune.custom_from_mask(
            model.encoder.layers[ii].fc1, name='weight', mask=
            mask_dict['encoder.layers.' + str(ii) + '.fc1.weight_mask']
        )
        model.encoder.layers[ii].fc1 = prune.custom_from_mask(
            model.encoder.layers[ii].fc1, name='bias', mask=
            mask_dict['encoder.layers.' + str(ii) + '.fc1.bias_mask']
        )

        #parameters_to_prune.append(model.encoder.layers[ii].fc2)
        model.encoder.layers[ii].fc2 = prune.custom_from_mask(
            model.encoder.layers[ii].fc2, name='weight', mask=
            mask_dict['encoder.layers.' + str(ii) + '.fc2.weight_mask']
        )
        model.encoder.layers[ii].fc2 = prune.custom_from_mask(
            model.encoder.layers[ii].fc2, name='bias', mask=
            mask_dict['encoder.layers.' + str(ii) + '.fc2.bias_mask']
        )

    return model
