import torch
import os
import soundfile as sf
import numpy as np
from spec import compute_spectrogram

class LIDDataset(torch.utils.data.Dataset):
    def __init__(self, args, tsv_file, huggingface_processor, seq_len=None):
        'Initialization'

        lans = ['es', 'fr', 'it', 'ky', 'nl', 'ru', 'sv_SE', 'tr', 'tt', 'zh_TW']
        lan2id = {x:ii for ii, x in enumerate(lans)}
        print(lan2id)
        f = open(tsv_file)
        _buffer = f.readline().strip('\n')
        home_pth = _buffer
        self.features = []
        self.labels = []
        while _buffer:
            _buffer = f.readline().strip('\n')
            if not '/' in _buffer:
                break
            _buffer = _buffer.split()[0]
            flac_path = os.path.join(home_pth, _buffer)
            wav, samplerate = sf.read(flac_path)
            if args.use_spec: # use spectrogram
                logspec, n_frames = compute_spectrogram(wav, samplerate)
                logspec = logspec.t()
                self.features.append(logspec)
            else: # use waveform
                self.features.append(wav)
            lan = flac_path.split('/')[8]
            self.labels.append(lan2id[lan])
        f.close()
        self.processor = huggingface_processor
        self.samplerate = samplerate
        if seq_len is not None:
            if args.use_spec: # use spectrogram
                self.seq_len = seq_len//160 + 1
            else:
                self.seq_len = int(seq_len)
        else:
            self.seq_len = seq_len
        self.args = args

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        wav = self.features[index]
        input_tensors = self.processor(wav, return_tensorzs="pt", sampling_rate=self.samplerate).input_values
        input_tensors = torch.tensor(input_tensors).float().squeeze()
        lan_label = self.labels[index]

        true_len = len(input_tensors)
        if self.seq_len is not None:
            # when seq_len is specified
            if len(input_tensors) >= self.seq_len: # longer than seq_len ==> truncate
                pin = np.random.randint(0, len(input_tensors) - self.seq_len + 1)
                input_tensors = input_tensors[pin:pin+self.seq_len]
                true_len = len(input_tensors)
            else: # zero-padding
                input_tensors = self._zero_pad(input_tensors, self.seq_len)

        return input_tensors, lan_label, true_len

    def _zero_pad(self, tensor, target_len, pad_value=0):
        if self.args.use_spec:
            pad = tensor.new(target_len, tensor.shape[-1]).fill_(pad_value)
            pad[:len(tensor),:] = tensor
        else:
            pad = tensor.new(target_len).fill_(pad_value)
            pad[:len(tensor)] = tensor

        return pad


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.
    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).
    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.
    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.
    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
        With the reference tensor.
        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
        With the reference tensor and dimension indicator.
        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask
