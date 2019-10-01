import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .decoder_fn_lib import inference_loop, train_loop
from .model_utils import sample_gaussian, dynamic_rnn, get_bi_rnn_encode, get_rnncell


class CVAEModel(nn.Module):
    def __init__(self, data_config, model_config, vocab_class):
        super(CVAEModel, self).__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.vocab_class = vocab_class

        self.vocab = vocab_class.vocab
        self.rev_vocab = vocab_class.rev_vocab
        self.vocab_size = len(self.vocab)

        self.topic_vocab = vocab_class.topic_vocab
        self.topic_vocab_size = len(self.topic_vocab)

        self.da_vocab = vocab_class.dialog_act_vocab
        self.da_vocab_size = len(self.da_vocab)

        self.max_utt_len = data_config['max_utt_len']
        self.pad_id = self.rev_vocab['<pad>']
        self.go_id = self.rev_vocab['<s>']
        self.eos_id = self.rev_vocab['</s>']

        self.ctx_cell_size = model_config['ctx_cell_size']
        self.sent_cell_size = model_config['sent_cell_size']
        self.dec_cell_size = model_config['dec_cell_size']

        self.latent_size = model_config['latent_size']
        self.embed_size = model_config['embed_size']
        self.sent_type = model_config['sent_type']
        self.keep_prob = model_config['keep_prob']
        self.num_layer = model_config['num_layer']

        self.use_hcf = model_config['use_hcf']
        self.device = torch.device(model_config['device'])

        self.dec_keep_prob = model_config['dec_keep_prob']
        self.keep_prob = model_config['keep_prob']

        self.topic_embed_size = model_config["topic_embed_size"]
        self.topic_embedding = nn.Embedding(self.topic_vocab_size, self.topic_embed_size)

        self.da_size = model_config["da_size"]
        self.da_embed_size = model_config["da_embed_size"]
        self.da_embedding = nn.Embedding(self.da_vocab_size, self.da_embed_size)

        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.pad_id)
        self.word_embedding.from_pretrained(torch.FloatTensor(vocab_class.word2vec), padding_idx=self.pad_id)

        # only use bi-rnn cell
        if self.sent_type == 'bi-rnn':
            self.bi_sent_cell = get_rnncell('gru', self.embed_size, self.sent_cell_size,
                                            self.keep_prob, num_layer=1, bidirectional=True)
            input_embedding_size = output_embedding_size = self.sent_cell_size * 2
        else:
            raise ValueError("unk sent_type. in this case we only use bi-rnn type")

        # context_embedding + floor (602)
        joint_embedding_size = input_embedding_size + 2
        print('joint_embedding_size:', joint_embedding_size)

        # RNN for context
        self.enc_cell = get_rnncell(model_config['cell_type'], joint_embedding_size, self.ctx_cell_size,
                                    keep_prob=1.0, num_layer=self.num_layer)
        self.attribute_fc1 = nn.Sequential(
            nn.Linear(self.da_embed_size, model_config['da_hidden_size']),
            nn.Tanh())

        # 30 + (5 * 2) + 600 = 640
        cond_embedding_size = self.topic_embed_size + \
                              (2 * model_config['meta_embed_size']) + self.ctx_cell_size
        print('cond_embedding_size:', cond_embedding_size)

        # recog network 640 + 600 + 10 = 1250
        recog_input_size = cond_embedding_size + output_embedding_size
        if self.use_hcf:
            recog_input_size += self.da_embed_size
        print('recog_input_size:', recog_input_size)

        self.recog_mulogvar_net = nn.Linear(recog_input_size, self.latent_size * 2)

        # prior network
        self.prior_mulogvar_net = nn.Sequential(
            nn.Linear(cond_embedding_size, np.maximum(self.latent_size * 2, 100)),
            nn.Tanh(),
            nn.Linear(np.maximum(self.latent_size * 2, 100), self.latent_size * 2)
        )

        gen_input_size = cond_embedding_size + self.latent_size
        print('gen_input_size:', gen_input_size)

        # BOW loss
        self.bow_project = nn.Sequential(
            nn.Linear(gen_input_size, model_config['bow_hidden_size']),
            nn.Tanh(),
            nn.Dropout(1 - self.keep_prob),
            nn.Linear(model_config['bow_hidden_size'], self.vocab_size)
        )

        # Y loss
        if self.use_hcf:
            self.da_project = nn.Sequential(
                nn.Linear(gen_input_size, model_config['act_hidden_size']),
                nn.Tanh(),
                nn.Dropout(1 - self.keep_prob),
                nn.Linear(model_config['act_hidden_size'], model_config['da_size'])
            )
            dec_input_size = gen_input_size + self.da_embed_size
        else:
            dec_input_size = gen_input_size
        print('dec_input_size:', dec_input_size)

        # Decoder
        if self.num_layer > 1:
            self.dec_init_state_net = nn.ModuleList(
                [nn.Linear(dec_input_size, self.dec_cell_size) for _ in range(self.num_layer)]
            )
        else:
            self.dec_init_state_net = nn.Linear(dec_input_size, self.dec_cell_size)

        dec_input_embedding_size = self.embed_size
        if self.use_hcf:
            dec_input_embedding_size += model_config['da_hidden_size']
        self.dec_cell = get_rnncell(model_config['cell_type'], dec_input_embedding_size,
                                    self.dec_cell_size, model_config['keep_prob'],
                                    num_layer=self.num_layer)
        self.dec_cell_proj = nn.Linear(self.dec_cell_size, self.vocab_size)

    def forward(self, feed_dict):
        is_train = feed_dict["is_train"]
        is_train_multiple = feed_dict.get("is_train_multiple", False)
        is_test_multi_da = feed_dict.get("is_test_multi_da", True)
        num_samples = feed_dict["num_samples"]

        """
        vec_context, vec_context_lens, vec_floors, 
        topics, my_profiles, ot_profiles, vec_outs, vec_out_lens, vec_out_das
        """

        context_lens = feed_dict['context_lens'].to(self.device).squeeze()
        input_contexts = feed_dict['vec_context'].to(self.device)
        floors = feed_dict['vec_floors'].to(self.device)
        out_tok = feed_dict['vec_outs'].to(self.device)
        topics = feed_dict['topics'].to(self.device).squeeze()
        my_profile = feed_dict['my_profile'].to(self.device)
        ot_profile = feed_dict['ot_profile'].to(self.device)
        out_das = feed_dict['out_das'].to(self.device).squeeze()
        output_lens = feed_dict['out_lens'].to(self.device).squeeze()

        # loader 에서 바로 출력 했기 때문에 forward 에서 padding + tensor 변환을 거침

        local_batch_size = input_contexts.size(0)
        max_dialog_len = input_contexts.size(1)
        max_seq_len = input_contexts.size(-1)

        input_contexts = input_contexts.view(-1, max_seq_len)
        out_das = out_das.squeeze()
        topics = topics.squeeze()

        relation_embedded = self.topic_embedding(topics)
        # if self.config['use_hcf']:
        #    da_embedded = self.da_embedding(out_da)

        input_embedded = self.word_embedding(input_contexts)

        output_embedded = self.word_embedding(out_tok)

        # only use bi-rnn
        if self.sent_type == 'bi-rnn':
            input_embedding, sent_size = get_bi_rnn_encode(input_embedded, self.bi_sent_cell,
                                                           self.max_utt_len)
            output_embedding, _ = get_bi_rnn_encode(output_embedded, self.bi_sent_cell,
                                                    self.max_utt_len)
        else:
            raise ValueError("unk sent_type. select one in [bow, rnn, bi-rnn]")

        # reshape input into dialogs
        input_embedding = input_embedding.view(-1, max_dialog_len, sent_size)

        if self.keep_prob < 1.0:
            input_embedding = F.dropout(input_embedding, 1 - self.keep_prob, is_train)

        # floors are already converted as one-hot
        floor_one_hot = floors.new_zeros((floors.numel(), 2), dtype=torch.float)
        floor_one_hot.data.scatter_(1, floors.view(-1, 1), 1)
        floor_one_hot = floor_one_hot.view(-1, max_dialog_len, 2)

        joint_embedding = torch.cat([input_embedding, floor_one_hot], 2)

        # contextRNN
        _, enc_last_state = dynamic_rnn(self.enc_cell, joint_embedding,
                                        sequence_length=context_lens,
                                        max_len=self.max_utt_len)
        if self.num_layer > 1:
            enc_last_state = torch.cat([_ for _ in torch.unbind(enc_last_state)], 1)
        else:
            enc_last_state = enc_last_state.squeeze(0)

        if self.use_hcf:
            attribute_embedding = self.da_embedding(out_das)
            attribute_fc1 = self.attribute_fc1(attribute_embedding)

            ctrl_attribute_embeddings = {
                da: self.da_embedding(torch.ones(local_batch_size, dtype=torch.long, device=self.device) * idx)
                for idx, da in enumerate(self.da_vocab)}
            ctrl_attribute_fc1 = {k: self.attribute_fc1(v) for (k, v) in ctrl_attribute_embeddings.items()}

        # 30, 5, 5, 600 // 640
        cond_list = [relation_embedded, my_profile, ot_profile, enc_last_state]
        cond_embedding = torch.cat(cond_list, 1)

        # recogntion network
        if self.use_hcf:
            recog_input = torch.cat([cond_embedding, output_embedding, attribute_fc1], 1)
            if is_train_multiple:
                ctrl_recog_inputs = {k: torch.cat([cond_embedding, output_embedding, v], 1) for (k, v) in
                                     ctrl_attribute_fc1.items()}
        else:
            recog_input = torch.cat([cond_embedding, output_embedding], 1)

        recog_mulogvar = self.recog_mulogvar_net(recog_input)
        recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)

        if is_train_multiple:
            ctrl_recog_mulogvars = {k: self.recog_mulogvar_net(v) for (k, v) in ctrl_recog_inputs.items()}
            ctrl_recog_mus = {}
            ctrl_recog_logvars = {}
            for k in ctrl_recog_mulogvars.keys():
                ctrl_recog_mus[k], ctrl_recog_logvars[k] = torch.chunk(ctrl_recog_mulogvars[k], 2, 1)

        # prior network
        prior_mulogvar = self.prior_mulogvar_net(cond_embedding)
        prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)

        # use sampled Z or posterior Z
        if not is_train:
            latent_samples = [sample_gaussian(prior_mu, prior_logvar) for _ in range(num_samples)]
        else:
            if not is_train_multiple:
                latent_samples = [sample_gaussian(recog_mu, recog_logvar)]
            if is_train_multiple:
                latent_samples = [sample_gaussian(recog_mu, recog_logvar) for _ in range(num_samples)]
                ctrl_latent_samples = {k: sample_gaussian(ctrl_recog_mus[k], ctrl_recog_logvars[k]) for k in
                                       ctrl_recog_mulogvars.keys()}
                ctrl_gen_inputs = {k: torch.cat([cond_embedding, v], 1) for (k, v) in ctrl_latent_samples.items()}

        gen_inputs = [torch.cat([cond_embedding, latent_sample], 1) for latent_sample in latent_samples]

        bow_logit = self.bow_project(gen_inputs[0])

        if self.use_hcf:

            da_logits = [self.da_project(gen_input) for gen_input in gen_inputs]
            da_probs = [F.softmax(da_logit, dim=1) for da_logit in da_logits]
            pred_attribute_embeddings = [torch.matmul(da_prob, self.da_embedding.weight) for da_prob in da_probs]

            if not is_train:
                selected_attr_embedding = pred_attribute_embeddings
                dec_inputs = [torch.cat((gen_input, selected_attr_embedding[i]), 1) for i, gen_input in
                              enumerate(gen_inputs)]
            else:
                selected_attr_embedding = attribute_embedding
                dec_inputs = [torch.cat((gen_input, selected_attr_embedding), 1) for gen_input in gen_inputs]
                if is_train_multiple:
                    ctrl_dec_inputs = {k: torch.cat((ctrl_gen_inputs[k], ctrl_attribute_embeddings[k]), 1) for k in
                                       ctrl_gen_inputs.keys()}
        else:
            da_logits = [gen_input.new_zeros(local_batch_size, self.da_size) for gen_input in gen_inputs]
            dec_inputs = [gen_input for gen_input in gen_inputs]

        # decoder

        if self.num_layer > 1:
            dec_init_states = [[self.dec_init_state_net[i](dec_input) for i in range(self.num_layer)] for dec_input in
                               dec_inputs]
            dec_init_states = [torch.stack(dec_init_state) for dec_init_state in dec_init_states]
            if is_train and is_train_multiple:
                ctrl_dec_init_states = {}
                for k, v in ctrl_dec_inputs.items():
                    ctrl_dec_init_states[k] = [self.dec_init_state_net[i](v) for i in range(self.num_layer)]
        else:
            dec_init_states = [self.dec_init_state_net(dec_input).unsqueeze(0) for dec_input in dec_inputs]
            if is_train and is_train_multiple:
                ctrl_dec_init_states = {k: self.dec_init_state_net(v).unsqueeze(0) for (k, v) in
                                        ctrl_dec_inputs.items()}

        dec_outss = []
        ctrl_dec_outs = {}
        if not is_train:
            dec_outs, _, final_ctx_state = inference_loop(self.dec_cell,
                                                          self.dec_cell_proj,
                                                          self.word_embedding,
                                                          encoder_state=dec_init_states[0],
                                                          start_of_sequence_id=self.go_id,
                                                          end_of_sequence_id=self.eos_id,
                                                          maximum_length=self.max_utt_len,
                                                          num_decoder_symbols=self.vocab_size,
                                                          context_vector=selected_attr_embedding[0],
                                                          decode_type='greedy')
            for i in range(1, num_samples):
                temp_outs, _, _ = inference_loop(self.dec_cell,
                                                 self.dec_cell_proj,
                                                 self.word_embedding,
                                                 encoder_state=dec_init_states[i],
                                                 start_of_sequence_id=self.go_id,
                                                 end_of_sequence_id=self.eos_id,
                                                 maximum_length=self.max_utt_len,
                                                 num_decoder_symbols=self.vocab_size,
                                                 context_vector=selected_attr_embedding[i],
                                                 decode_type='greedy')
                dec_outss.append(temp_outs)
            if is_test_multi_da:
                for key, value in ctrl_attribute_embeddings.items():
                    ctrl_dec_outs[key], _, _ = inference_loop(self.dec_cell,
                                                              self.dec_cell_proj,
                                                              self.word_embedding,
                                                              encoder_state=dec_init_states[0],
                                                              start_of_sequence_id=self.go_id,
                                                              end_of_sequence_id=self.eos_id,
                                                              maximum_length=self.max_utt_len,
                                                              num_decoder_symbols=self.vocab_size,
                                                              context_vector=value,
                                                              decode_type='greedy')

        else:
            # remove eos token
            input_tokens = out_tok[:, :-1].clone()
            input_tokens[input_tokens == self.eos_id] = 0
            if self.dec_keep_prob < 1.0:
                keep_mask = input_tokens.new_empty(input_tokens.size()).bernoulli_(self.dec_keep_prob)
                input_tokens = input_tokens * keep_mask
            dec_input_embedded = self.word_embedding(input_tokens)
            dec_seq_len = output_lens - 1

            dec_input_embedded = F.dropout(dec_input_embedded, 1 - self.keep_prob, is_train)
            dec_outs, _, final_ctx_state = train_loop(self.dec_cell,
                                                      self.dec_cell_proj,
                                                      dec_input_embedded,
                                                      init_state=dec_init_states[0],
                                                      context_vector=selected_attr_embedding,
                                                      sequence_length=dec_seq_len,
                                                      max_len=self.max_utt_len - 1)

            if is_train_multiple:
                for i in range(1, num_samples):
                    temp_outs, _, _ = inference_loop(self.dec_cell,
                                                     self.dec_cell_proj,
                                                     self.word_embedding,
                                                     encoder_state=dec_init_states[i],
                                                     start_of_sequence_id=self.go_id,
                                                     end_of_sequence_id=self.eos_id,
                                                     maximum_length=self.max_utt_len,
                                                     num_decoder_symbols=self.vocab_size,
                                                     context_vector=selected_attr_embedding,
                                                     decode_type='greedy')
                    dec_outss.append(temp_outs)
                for key, value in ctrl_attribute_embeddings.items():
                    ctrl_dec_outs[key], _, _ = inference_loop(self.dec_cell,
                                                              self.dec_cell_proj,
                                                              self.word_embedding,
                                                              encoder_state=ctrl_dec_init_states[key],
                                                              start_of_sequence_id=self.go_id,
                                                              end_of_sequence_id=self.eos_id,
                                                              maximum_length=self.max_utt_len,
                                                              num_decoder_symbols=self.vocab_size,
                                                              context_vector=value,
                                                              decode_type='greedy')

        model_output = {"dec_out": dec_outs, "dec_outss": dec_outss, "ctrl_dec_out": ctrl_dec_outs,
                        "final_ctx_state": final_ctx_state, "bow_logit": bow_logit, "da_logit": da_logits[0],
                        "out_token": out_tok, "out_das": out_das, "recog_mulogvar": recog_mulogvar,
                        "prior_mulogvar": prior_mulogvar}

        output_sents, ctrl_output_sents, sampled_output_sents, output_logits, real_output_sents, real_output_logits, input_context_sents = \
            self.index2sent(feed_dict['vec_context'], context_lens, model_output)
        model_output["output_sents"] = output_sents
        model_output["ctrl_output_sents"] = ctrl_output_sents
        model_output["sampled_output_sents"] = sampled_output_sents
        model_output["output_das"] = output_logits
        model_output["real_output_sents"] = real_output_sents
        model_output["real_output_das"] = real_output_logits
        model_output["context_sents"] = input_context_sents

        return model_output

    def index2sent(self, input_contexts, context_lens, model_output):
        # shape of dec_out_would be (Batch size, sentence len, size
        dec_out = model_output["dec_out"].cpu()
        dec_outss = [e.cpu() for e in model_output["dec_outss"]]
        ctrl_dec_out = model_output["ctrl_dec_out"]
        ctrl_dec_out = {k: v.cpu() for (k, v) in ctrl_dec_out.items()}
        da_logits = model_output["da_logit"].cpu()
        real_outputs = model_output["out_token"].cpu()
        real_das = model_output["out_das"].cpu()
        input_contexts = input_contexts.cpu()
        context_lens = context_lens.cpu()

        input_context_sents = []
        for batch_id, target_context in enumerate(input_contexts):
            input_context_sent = []
            context_len = context_lens[batch_id]
            valid_contexts = target_context[:context_len]
            for sent_id, context_sent in enumerate(valid_contexts):
                sent = []
                for dec_index in context_sent:
                    if dec_index == self.eos_id:
                        sent.append("</s>")
                        break
                    else:
                        word = self.vocab[dec_index]
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
                if dec_index == self.eos_id:
                    sent.append("</s>")
                    break
                else:
                    word = self.vocab[dec_index]
                    sent.append(word)
            output_sentences.append(sent)

        for k in ctrl_dec_indice_sents.keys():
            for dec_indice in ctrl_dec_indice_sents[k]:
                sent = []
                for dec_index in dec_indice:
                    if dec_index == self.eos_id:
                        sent.append("</s>")
                        break
                    else:
                        word = self.vocab[dec_index]
                        sent.append(word)
                ctrl_output_sents[k].append(sent)

        for i, sampled_dec_indice_sent in enumerate(sampled_dec_indice_sents):
            for dec_indice in sampled_dec_indice_sent:
                sent = []
                for dec_index in dec_indice:
                    if dec_index == self.eos_id:
                        sent.append("</s>")
                        break
                    else:
                        word = self.vocab[dec_index]
                        sent.append(word)
                sampled_output_sents[i].append(sent)

        for da_logit in da_logits:
            da_word = self.da_vocab[da_logit]
            output_logits.append(da_word)

        for real_output in real_outputs:
            real_sent = []
            for real_out_index in real_output:
                if real_out_index == self.eos_id:
                    real_sent.append("</s>")
                    break
                else:
                    word = self.vocab[real_out_index]
                    real_sent.append(word)
            real_output_sents.append(real_sent)

        for real_da_logit in real_das:
            da_word = self.da_vocab[real_da_logit]
            real_output_logits.append(da_word)

        return output_sentences, ctrl_output_sents, sampled_output_sents, output_logits, \
               real_output_sents, real_output_logits, \
               input_context_sents
