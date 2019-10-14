"""
Copyright 2019 Pingpong AI Research, ScatterLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import torch


def index2sent(input_contexts, context_lens, model_output, feed_real, eos_id, vocab, da_vocab):
    """
    Interpret model_output to human readable sentences.

    :param input_contexts: input context (from 0 to previous turn) vector.
    :param context_lens: lengths of each input context sentence.
    :param model_output: target output to be generated.
    :param feed_real: if True, the function will also print out real sentences.
    :param eos_id: end of sequence (<eos>) token index.
    :param vocab: vocab list.
    :param da_vocab: dialog act word list.
    :return:
    """
    dec_out = model_output["dec_out"].cpu()
    dec_outss = [e.cpu() for e in model_output["dec_outss"]]
    ctrl_dec_out = model_output["ctrl_dec_out"]
    ctrl_dec_out = {k: v.cpu() for (k, v) in ctrl_dec_out.items()}
    da_logits = model_output["da_logit"].cpu()
    input_contexts = input_contexts.cpu()
    context_lens = context_lens.cpu()

    if feed_real:
        real_outputs = model_output["out_token"].cpu()
        real_das = model_output["out_das"].cpu()
    else:
        real_outputs = []
        real_das = []

    input_context_sents = []
    for batch_id, target_context in enumerate(input_contexts):
        input_context_sent = []
        context_len = context_lens[batch_id]
        valid_contexts = target_context[:context_len]
        for sent_id, context_sent in enumerate(valid_contexts):
            sent = []
            for dec_index in context_sent:
                if dec_index == eos_id:
                    sent.append("</s>")
                    break
                else:
                    word = vocab[dec_index]
                    sent.append(word)
            input_context_sent.append(sent)
        input_context_sents.append(input_context_sent)

    output_sentences = []
    real_output_sents = []
    output_logits = []
    real_output_logits = []
    ctrl_output_sents = {k: [] for k in ctrl_dec_out.keys()}
    sampled_output_sents = [[] for _ in range(len(dec_outss))]

    _, dec_indice_sents = torch.max(dec_out, 2)
    ctrl_dec_indice_sents = {}
    for k in ctrl_dec_out.keys():
        _, ctrl_dec_indice_sents[k] = torch.max(ctrl_dec_out[k], 2)
    sampled_dec_indice_sents = []
    for e in dec_outss:
        _, temp_dec_indice_sents = torch.max(e, 2)
        sampled_dec_indice_sents.append(temp_dec_indice_sents)
    _, da_logits = torch.max(da_logits, 1)

    dec_indice_sents = dec_indice_sents.numpy()
    ctrl_dec_indice_sents = {k: v.numpy() for (k, v) in ctrl_dec_indice_sents.items()}
    sampled_dec_indice_sents = [e.numpy() for e in sampled_dec_indice_sents]
    da_logits = da_logits.numpy()

    for dec_indice in dec_indice_sents:
        sent = []
        for dec_index in dec_indice:
            if dec_index == eos_id:
                sent.append("</s>")
                break
            else:
                word = vocab[dec_index]
                sent.append(word)
        output_sentences.append(sent)

    for k in ctrl_dec_indice_sents.keys():
        for dec_indice in ctrl_dec_indice_sents[k]:
            sent = []
            for dec_index in dec_indice:
                if dec_index == eos_id:
                    sent.append("</s>")
                    break
                else:
                    word = vocab[dec_index]
                    sent.append(word)
            ctrl_output_sents[k].append(sent)

    for i, sampled_dec_indice_sent in enumerate(sampled_dec_indice_sents):
        for dec_indice in sampled_dec_indice_sent:
            sent = []
            for dec_index in dec_indice:
                if dec_index == eos_id:
                    sent.append("</s>")
                    break
                else:
                    word = vocab[dec_index]
                    sent.append(word)
            sampled_output_sents[i].append(sent)

    for da_logit in da_logits:
        da_word = da_vocab[da_logit]
        output_logits.append(da_word)

    if feed_real:
        for real_output in real_outputs:
            real_sent = []
            for real_out_index in real_output:
                if real_out_index == eos_id:
                    real_sent.append("</s>")
                    break
                else:
                    word = vocab[real_out_index]
                    real_sent.append(word)
            real_output_sents.append(real_sent)

        for real_da_logit in real_das:
            da_word = da_vocab[real_da_logit]
            real_output_logits.append(da_word)

    return output_sentences, ctrl_output_sents, sampled_output_sents, output_logits, \
           real_output_sents, real_output_logits, \
           input_context_sents