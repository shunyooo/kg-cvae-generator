import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import json
import pickle
import torch
import torch.nn as nn


def load_pickle_class(path):
    with open(path, 'rb') as file:
        dv = pickle.load(file)
    return dv


def save_pickle_class(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)
    print('save this class in', path)


def load_dict_from_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        output = json.load(file)
    return output


def save_dict_as_json(path, target):
    with open(path, 'w') as fp:
        json.dump(target, fp)
    print('target', len(target), 'dict is saved in', path)


def get_bleu_stats(ref, hyps):
    scores = []
    for hyp in hyps:
        try:
            scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                        weights=[1. / 3, 1. / 3, 1. / 3]))
        except:
            scores.append(0.0)
    return np.max(scores), np.mean(scores)


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                           - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                           - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
    return kld


def norm_log_liklihood(x, mu, logvar):
    return -0.5 * torch.sum(logvar + np.log(2 * np.pi) + torch.div(torch.pow((x - mu), 2), torch.exp(logvar)), 1)


def sample_gaussian(mu, logvar):
    epsilon = logvar.new_empty(logvar.size()).normal_()
    std = torch.exp(0.5 * logvar)
    z = mu + std * epsilon
    return z


def dynamic_rnn(cell, inputs, sequence_length, max_len, init_state=None, output_fn=None):
    sorted_lens, len_ix = sequence_length.sort(0, descending=True)

    # Used for later reorder
    inv_ix = len_ix.clone()
    inv_ix[len_ix] = torch.arange(0, len(len_ix)).type_as(inv_ix)

    # The number of inputs that have lengths > 0
    valid_num = torch.sign(sorted_lens).long().sum().item()
    zero_num = inputs.size(0) - valid_num

    sorted_inputs = inputs[len_ix].contiguous()
    if init_state is not None:
        sorted_init_state = init_state[:, len_ix].contiguous()

    packed_inputs = pack_padded_sequence(sorted_inputs[:valid_num], list(sorted_lens[:valid_num]),
                                         batch_first=True)

    if init_state is not None:
        outputs, state = cell(packed_inputs, sorted_init_state[:, :valid_num])
    else:
        outputs, state = cell(packed_inputs)

    # Reshape *final* output to (batch_size, hidden_size)
    outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=max_len)

    # Add back the zero lengths
    if zero_num > 0:
        outputs = torch.cat(
            [outputs, outputs.new_zeros(zero_num, outputs.size(1), outputs.size(2))], 0)
        if init_state is not None:
            state = torch.cat([state, sorted_init_state[:, valid_num:]], 1)
        else:
            state = torch.cat([state, state.new_zeros(state.size(0), zero_num, state.size(2))], 1)

    # Reorder to the original order
    new_outputs = outputs[inv_ix].contiguous()
    new_state = state[:, inv_ix].contiguous()

    # compensate the last last layer dropout, necessary????????? need to check!!!!!!!!
    new_new_state = F.dropout(new_state, cell.dropout, cell.training)
    new_new_outputs = F.dropout(new_outputs, cell.dropout, cell.training)

    if output_fn is not None:
        new_new_outputs = output_fn(new_new_outputs)

    return new_new_outputs, new_new_state


"""

    sorted_lens, len_ix = sequence_length.sort(0, descending=True)

    print(sorted_lens)
    print(len_ix)
    # Used for later reorder
    inv_ix = len_ix.clone()
    inv_ix[len_ix] = torch.arange(0, len(len_ix)).type_as(inv_ix)

    # The number of inputs that have lengths > 0
    valid_num = torch.sign(sorted_lens).long().sum().item()
    zero_num = inputs.size(0) - valid_num
    # print('zero_num:', zero_num)

    sorted_inputs = inputs[len_ix].contiguous()
    if init_state is not None:
        sorted_init_state = init_state[:, len_ix].contiguous()

    packed_inputs = pack_padded_sequence(sorted_inputs[:valid_num], list(sorted_lens[:valid_num]), batch_first=True)

    if init_state is not None:
        outputs, state = cell(packed_inputs, sorted_init_state[:, :valid_num])
    else:
        outputs, state = cell(packed_inputs)

    # Reshape *final* output to (batch_size, hidden_size)
    outputs, _ = pad_packed_sequence(outputs, batch_first=True)

    # Add back the zero lengths
    if zero_num > 0:
        outputs = torch.cat([outputs, outputs.new_zeros(zero_num, outputs.size(1), outputs.size(2))], 0)
        if init_state is not None:
            state = torch.cat([state, sorted_init_state[:, valid_num:]], 1)
        else:
            state = torch.cat([state, state.new_zeros(state.size(0), zero_num, state.size(2))], 1)

    # Reorder to the original order
    outputs = outputs[inv_ix].contiguous()
    state = state[:, inv_ix].contiguous()

    # compensate the last last layer dropout, necessary????????? need to check!!!!!!!!
    state = F.dropout(state, cell.dropout, cell.training)
    outputs = F.dropout(outputs, cell.dropout, cell.training)

    if output_fn is not None:
        outputs = output_fn(outputs)

    return outputs, state
    """


def get_bow(embedding, avg=False):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    """
    embedding_size = embedding.size(2)
    if avg:
        return embedding.mean(1), embedding_size
    else:
        return embedding.sum(1), embedding_size


def get_rnn_encode(embedding, cell, length_mask=None):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    The padding should have zero
    """
    if length_mask is None:
        length_mask = torch.sum(torch.sign(torch.max(torch.abs(embedding), 2)[0]), 1)
        length_mask = length_mask.long()
    _, encoded_input = dynamic_rnn(cell, embedding, sequence_length=length_mask)

    # get only the last layer
    encoded_input = encoded_input[-1]
    return encoded_input, cell.hidden_size


def get_bi_rnn_encode(embedding, cell, max_len, length_mask=None):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    The padding should have zero
    """
    if length_mask is None:
        length_mask = torch.sum(torch.sign(torch.max(torch.abs(embedding), 2)[0]), 1)
        length_mask = length_mask.long()
    _, encoded_input = dynamic_rnn(cell, embedding, sequence_length=length_mask, max_len=max_len)
    # get only the last layer
    encoded_input2 = torch.cat([encoded_input[-2], encoded_input[-1]], 1)
    return encoded_input2, cell.hidden_size * 2


def get_rnncell(cell_type, input_size, cell_size, keep_prob, num_layer, bidirectional=False):
    if cell_type == 'gru':
        cell = nn.GRU(input_size, cell_size, num_layers=num_layer, dropout=1 - keep_prob,
                      bidirectional=bidirectional, batch_first=True)
    elif cell_type == 'lstm':
        cell = nn.LSTM(input_size, cell_size, num_layers=num_layer, dropout=1 - keep_prob,
                       bidirectional=bidirectional, batch_first=True)
    else:
        raise ValueError("Set cell type RNN: 'gru' or 'lstm'")
    return cell


def collate_func(batch):
    def sentiment_real2cat(sent):
        thr = [-0.5, -0.35, 0.35, 0.5]
        return sum([sent > t for t in thr])

    dialog, meta, response, output_sentiments, rules = list(zip(*batch))
    ctx_length = [[len(xx) for xx in x] for x in dialog]
    rules = [[[1, 0] if xx == 'SEND' else [0, 1] for xx in x] for x in rules]
    my_profile = list()
    ot_profile = list()
    relations = list()
    for m in meta:
        my_profile.append(sum([a * b for a, b in zip(m[0:2], [3, 1])]))
        ot_profile.append(sum([a * b for a, b in zip(m[2:4], [3, 1])]))
        relations.append(m[-1])

    output_sentiments_cat = sentiment_real2cat(output_sentiments)

    response_length = [len(x) for x in response]

    feeding = {'input_contexts': dialog,
               'context_lens': ctx_length,
               'floors': rules,
               'relations': relations,
               'my_profile': my_profile,
               'ot_profile': ot_profile,
               'output_tokens': response,
               'output_sentiments': output_sentiments_cat,
               'output_lens': response_length}

    return feeding
