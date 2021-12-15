import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import distributions
import os
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from lid_modules import resnet34, resnet50
import itertools

def load_models_and_criterions(
    args, filename, arg_overrides=None, task=None, model_state=None
):

    if arg_overrides is None:
        arg_overrides = {}

    arg_overrides["wer_args"] = None

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
        task = tasks.setup_task(cfg.task)

    model = task.build_model(cfg.model)

    if args.xlsr_pretrained and not args.debug_mode:
        print('loading pretrained xlsr...')
        model.load_state_dict(state["model"], strict=True)
    else:
        print('xlsr is not pretrained')

    criterion = task.build_criterion(cfg.criterion)
    if "criterion" in state:
        criterion.load_state_dict(state["criterion"], strict=True)

    return model


class LID_XLSR53(nn.Module):
    """ XLSR53 + language_tempalte + LID
    """

    def __init__(self, args, logger):
        super(LID_XLSR53, self).__init__()
        path = args.path; template_num = args.lan_codebooks; embed_dim = args.lan_codebook_dim
        logger.info("| loading model(s) from {}".format(path))
        if args.use_spec:
            self.resnet = resnet34()
            self.reduce_dim  = nn.Linear(128, embed_dim)
            self.stl = STLv2(args.straight_through, template_num=template_num, embed_dim=embed_dim)
            if args.lan_residual:
                self.fc  = nn.Linear(embed_dim*(template_num+1), 10)
                self.bn  = nn.BatchNorm1d(embed_dim*(template_num+1))
            else:
                self.fc  = nn.Linear(embed_dim*template_num, 10)
                self.bn  = nn.BatchNorm1d(embed_dim*template_num)
        else:
            self.xlsr53 = load_models_and_criterions(args, path)
            self.reduce_dim  = nn.Linear(1024, embed_dim)
            self.stl = STL(template_num=template_num, embed_dim=embed_dim)
            if args.lan_residual:
                self.fc  = nn.Linear(embed_dim*2, 10)
                self.bn  = nn.BatchNorm1d(embed_dim*2)
            else:
                self.fc  = nn.Linear(embed_dim, 10)
                self.bn  = nn.BatchNorm1d(embed_dim)

        self.args = args

    def forward(self, x, padding_mask):
        if self.args.use_spec: # spec --> resnet
            #print(padding_mask.shape) # torch.Size([1, 296])
            x = self.resnet(x) # torch.Size([1, 37, 128])
            #print(x.shape)
            x = self._mean_pool(x, padding_mask, downsampling_rate=1) # torch.Size([1, 128])
            #print(x.shape)
        else: # raw waveform --> xlsr53
            #print(x.shape) # B, T
            #print(padding_mask.shape)
            x = self.xlsr53.extract_features(x, padding_mask=padding_mask)[0]
            #print(x.shape) # B, T/320, 1024
            x = self._mean_pool(x, padding_mask)
            #print(x.shape) # B, 1024
        x = self.reduce_dim(x) # no non-linearity afterwards
        style_embeddings, _, attention_score = self.stl(x)
        #print(style_embeddings.shape)
        style_embeddings = style_embeddings.squeeze(1)
        #print(torch.cat((x, style_embeddings), dim=-1).shape)
        if self.args.lan_residual:
            x = self.fc(F.relu(self.bn(torch.cat((x, style_embeddings), dim=-1))))
        else:
            x = self.fc(F.relu(self.bn(style_embeddings)))

        return F.log_softmax(x, dim=-1), attention_score, self.attention_diversity_measure(attention_score)

    def extract_lan_embed(self, x, padding_mask):
        if self.args.use_spec: # spec --> resnet
            #print(padding_mask.shape) # torch.Size([1, 296])
            x = self.resnet(x) # torch.Size([1, 37, 128])
            #print(x.shape)
            x = self._mean_pool(x, padding_mask, downsampling_rate=1) # torch.Size([1, 128])
            #print(x.shape)
        else: # raw waveform --> xlsr53
            #print(x.shape) # B, T
            #print(padding_mask.shape)
            x = self.xlsr53.extract_features(x, padding_mask=padding_mask)[0]
            #print(x.shape) # B, T/320, 1024
            x = self._mean_pool(x, padding_mask)
            #print(x.shape) # B, 1024
        x = self.reduce_dim(x) # no non-linearity afterwards
        style_embeddings, _, attention_score = self.stl(x)
        #print(style_embeddings.shape)
        style_embeddings = style_embeddings.squeeze(1)
        #print(torch.cat((x, style_embeddings), dim=-1).shape)
        if self.args.lan_residual:
            return torch.cat((x, style_embeddings), dim=-1)

    def attention_diversity_measure(self, attention_score):
        """ how diverse/evenly distributed the attention scores are over codebooks
        (higher the more diverse/evenly distributed)
        """
        #score = distributions.Categorical(probs=attention_score).entropy()
        attention_score = attention_score.squeeze() # B, num_codes
        diversity = torch.mean(attention_score * torch.log(attention_score + 1e-7), dim=0) # take mean over batch dim
        diversity_score = torch.exp(-diversity).sum() # sum over codebooks
        #print(attention_score, diversity_score)

        #dist.Categorical(probs=((x2-x1) + 1e-7)).entropy() # the lower, the more difference x2 and x2 dists are

        return diversity_score

    def inter_lan_dist_entropy(self, attention_score, targets):
        """ how diverse/evenly distributed the attention scores are over aross languages
        (lower the more diverse/evenly distributed)
        """
        #print(attention_score.shape) # B, codes
        #print(targets.shape) # B
        assert attention_score.shape[0] == targets.shape[0]

        # 1. get lan-wise attention_score
        lans = ['es', 'fr', 'it', 'ky', 'nl', 'ru', 'sv_SE', 'tr', 'tt', 'zh_TW']
        batchlan2attention_score = {x:torch.zeros(self.args.lan_codebooks).cuda() for ii, x in enumerate(lans)}
        lan2cnt = {x:0 for ii, x in enumerate(lans)}
        for ii in range(len(targets)):
            target_lan = lans[targets[ii].item()]
            batchlan2attention_score[target_lan] += torch.sum(attention_score[ii], dim=0).squeeze()
            lan2cnt[target_lan] += 1

        # 2. get avg attention_score for each lan
        avglan2attention_score = {x:torch.zeros(self.args.lan_codebooks).cuda() for ii, x in enumerate(lans)}
        for lan, attention_score in batchlan2attention_score.items():
            avglan2attention_score[lan] = attention_score / lan2cnt[lan]
        del batchlan2attention_score, lan2cnt

        # 3. compute dist difference between each lan's attention_score. go through all combination e.g. (A,B), (A,C), (A,D), etc
        inter_lan_dist_entropy = 0
        for lan_combo in list(itertools.combinations(lans, 2)):
            dist1 = avglan2attention_score[lan_combo[0]]; dist2 = avglan2attention_score[lan_combo[1]]
            inter_lan_dist_entropy += distributions.Categorical(probs=((dist1 - dist2) + 1e-7)).entropy() # the lower, the more different two dists are
        return inter_lan_dist_entropy


    def retrieve_lan_templates(self, x, padding_mask):
        x = self.xlsr53.extract_features(x, padding_mask=padding_mask)[0]
        #print(x.shape) # B, T/320, 1024
        x = self._mean_pool(x, padding_mask)
        #print(x.shape) # B, 1024
        x = self.reduce_dim(x) # no non-linearity afterwards
        _, lan_templates, attention_score = self.stl(x)

        return lan_templates, self.attention_diversity_measure(attention_score)

    def _mean_pool(self, x, padding_mask, downsampling_rate=320):
        """ pool across time dimension with masking take-cared
        """
        non_padding_mask = ~padding_mask
        mean_x = x.new(x.shape[0], x.shape[2]).fill_(0)
        for ii, sample in enumerate(x):
            T_len = torch.sum(non_padding_mask[ii]) / downsampling_rate
            mean_x[ii] = torch.sum(sample, dim=0)/T_len.item()
        return mean_x

class STLv2(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, straight_through=False, template_num=4, embed_dim=128):

        super().__init__()
        atten_heads = 1
        self.banks = nn.Parameter(torch.FloatTensor(template_num, embed_dim))
        self.embed_dim = embed_dim
        init.normal_(self.banks, mean=0, std=0.5)
        self.straight_through = straight_through

    def forward(self, inputs):
        B = inputs.shape[0]
        #template_banks = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, template_num, E // num_heads]
        similarity = F.sigmoid(torch.matmul(inputs, self.banks.t())) # B, template_num
        #print(similarity.shape)

        if self.straight_through:
            similarity_round = torch.round(similarity)
            #print(similarity_round.shape)
            similarity = similarity - (similarity - similarity_round).detach() # straight through estimator
            #print(inputs.shape, similarity.shape) # torch.Size([512, 32]) torch.Size([512, 8])
            #x = x - (y+x).detach() x = x - (x-y).detach()

        similarity_rep = similarity.unsqueeze(-1).repeat(1, 1, self.embed_dim)  # B, template_num, embed_dim
        concat_weighted_banks = (similarity_rep * self.banks).view(B, -1) # B, template_num*embed_dim
        #print(concat_weighted_banks.shape)

        return concat_weighted_banks, self.banks, similarity

class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, template_num=4, embed_dim=128):

        super().__init__()
        atten_heads = 1
        self.embed = nn.Parameter(torch.FloatTensor(template_num, embed_dim // atten_heads))
        d_q = embed_dim
        d_k = embed_dim // atten_heads
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=embed_dim, num_heads=atten_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        template_banks = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, template_num, E // num_heads]
        #print(template_banks.shape) # B, num_temlates, embedd size
        style_embed, attention_score = self.attention(query, template_banks)

        return style_embed, template_banks, attention_score

class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)
        #print(scores.shape) # B, 1, 1, num_templates

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out, scores


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

