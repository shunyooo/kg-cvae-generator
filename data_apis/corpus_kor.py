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


from data_apis.corpus import CVAECorpus
from collections import Counter
import numpy as np
import os

from gensim.models.wrappers import FastText


class PingpongDialogCorpus(CVAECorpus):
    def __init__(self, config):
        self.reserved_token_for_dialog = ['<s>', '<d>', '</s>']
        self.reserved_token_for_gen = ['<pad>', '<unk>', '<sos>', '<eos>']
        super(PingpongDialogCorpus, self).__init__(config)

    def process(self, json_data):
        """new_dialog: [(a, 1/0), (a,1/0)], new_meta: (a, b, topic), new_utt: [[a,b,c)"""
        """ 1 is own utt and 0 is other's utt"""
        new_dialog = []
        new_meta = []
        new_utts = []
        bod_utt = ["<s>", "<d>", "</s>"]
        all_lenes = []

        for session_data in json_data:
            session_utts = session_data['utts']
            a_info = session_data['A']
            b_info = session_data['B']

            lower_utts = [(caller, tokenized_sent, senti_label)
                          for caller, tokenized_sent, raw_sent, _, senti_label in session_utts]
            all_lenes.extend([len(u) for c, u, f in lower_utts])

            vec_a_age_meta = [0, 0, 0]
            vec_a_age_meta[a_info['age']] = 1
            vec_a_sex_meta = [0, 0]
            vec_a_sex_meta[a_info['sex']] = 1
            vec_a_meta = vec_a_age_meta + vec_a_sex_meta

            vec_b_age_meta = [0, 0, 0]
            vec_b_age_meta[b_info['age']] = 1
            vec_b_sex_meta = [0, 0]
            vec_b_sex_meta[b_info['sex']] = 1
            vec_b_meta = vec_b_age_meta + vec_b_sex_meta

            topic = session_data["topic"] + "_" + a_info['relation_group']

            meta = (vec_a_meta, vec_b_meta, topic)

            dialog = [(bod_utt, 0, None)] + \
                     [(utt, int(caller=="B"), senti) for caller, utt, senti in lower_utts]

            new_utts.extend([bod_utt] + [utt for caller, utt, senti in lower_utts])
            new_dialog.append(dialog)
            new_meta.append(meta)

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lenes), float(np.mean(all_lenes))))
        return new_dialog, new_meta, new_utts

    def build_vocab(self, max_vocab_cnt):
        all_words = []
        for tokens in self.train_corpus[self.utt_id]:
            all_words.extend(tokens)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus), len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1], float(discard_wc) / len(all_words)))

        self.vocab = self.reserved_token_for_gen + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab["<unk>"]
        print("<d> index %d" % self.rev_vocab["<d>"])

        # create topic vocab
        all_topics = []
        for a, b, topic in self.train_corpus[self.meta_id]:
            all_topics.append(topic)
        self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
        self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}
        print("%d topics in train data" % len(self.topic_vocab))

        # get dialog act labels
        all_sentiments = []
        for dialog in self.train_corpus[self.dialog_id]:
            all_sentiments.extend([senti for caller, utt, senti in dialog if senti is not None])
        # self.dialog_act_vocab = [t for t, cnt in Counter(all_sentiments).most_common()]
        self.dialog_act_vocab = ["StrongNeg", "WeakNeg", "Neutral", "WeakPos", "StrongPos"] # for da control in test
        self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}
        print(self.dialog_act_vocab)
        print("%d dialog acts in train data" % len(self.dialog_act_vocab))

    def load_word2vec(self):
        if not os.path.exists(self.word_vec_path):
            return
        raw_word2vec = FastText.load_fasttext_format(self.word_vec_path)

        self.word2vec = []
        reserved_tokens = self.reserved_token_for_dialog + self.reserved_token_for_gen
        oov_cnt = 0
        for v in self.vocab:
            if v == "<pad>":
                vec = np.zeros(self.word2vec_dim)
            elif v in reserved_tokens:
                vec = np.random.randn(self.word2vec_dim) * 0.1
            else:
                if v in raw_word2vec:
                    vec = raw_word2vec[v]
                else:
                    oov_cnt += 1
                    vec = np.random.randn(self.word2vec_dim) * 0.1
            self.word2vec.append(vec)
        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))
