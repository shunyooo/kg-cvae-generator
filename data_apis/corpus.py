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


import json
import os


class CVAECorpus(object):
    dialog_act_id = 0
    sentiment_id = 1
    liwc_id = 2

    def __init__(self, config):
        self.config = config
        self._path = config['data_dir']
        train_file_name = config['train_filename']
        test_file_name = config['test_filename']
        valid_file_name = config['valid_filename']

        train_file_path = os.path.join(self._path, train_file_name)
        test_file_path = os.path.join(self._path, test_file_name)
        valid_file_path = os.path.join(self._path, valid_file_name)

        self.word_vec_path = config['word2vec_path']
        self.word2vec_dim = config['embed_size']
        self.word2vec = None
        self.dialog_id = 0
        self.meta_id = 1
        self.utt_id = 2

        with open(train_file_path, "r") as train_file_reader:
            if train_file_name.endswith(".json"):
                self.train_corpus_data = json.load(train_file_reader)
            elif train_file_name.endswith(".jsonl"):
                jsonl_content = train_file_reader.read().strip()
                self.train_corpus_data = [json.loads(jline) for jline in jsonl_content.split('\n')]
            else:
                raise ValueError("Not supported file format for train data.")

        with open(test_file_path, "r") as test_file_reader:
            if test_file_name.endswith(".json"):
                self.test_corpus_data = json.load(test_file_reader)
            elif test_file_name.endswith(".jsonl"):
                jsonl_content = test_file_reader.read().strip()
                self.test_corpus_data = [json.loads(jline) for jline in jsonl_content.split('\n')]
            else:
                raise ValueError("Not supported file format for test data.")

        with open(valid_file_path, "r") as valid_file_reader:
            if valid_file_name.endswith(".json"):
                self.valid_corpus_data = json.load(valid_file_reader)
            elif valid_file_name.endswith(".jsonl"):
                jsonl_content = valid_file_reader.read().strip()
                self.valid_corpus_data = [json.loads(jline) for jline in jsonl_content.split('\n')]
            else:
                raise ValueError("Not supported file format for test data.")

        print("Start process train corpus...")
        self.train_corpus = self.process(self.train_corpus_data)
        print("Start process test corpus...")
        self.test_corpus = self.process(self.test_corpus_data)
        print("Start process valid corpus...")
        self.valid_corpus = self.process(self.valid_corpus_data)

        print("Start building vocab")
        self.build_vocab(config['max_vocab_count'])

        self.load_word2vec()
        print("Done loading corpus")

    def process(self, json_data):
        pass

    def build_vocab(self, max_vocab_cnt):
        pass

    def load_word2vec(self):
        pass

    def get_utt_corpus(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([self.rev_vocab.get(t, self.unk_id) for t in line])
            return results
        # convert the corpus into ID
        id_train = _to_id_corpus(self.train_corpus[self.utt_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.utt_id])
        id_test = _to_id_corpus(self.test_corpus[self.utt_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_dialog_corpus(self):
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                temp = []
                # convert utterance and feature into numeric numbers
                for utt, floor, feat in dialog:
                    if feat is not None:
                        id_feat = self.rev_dialog_act_vocab[feat]
                    else:
                        id_feat = None
                    temp.append(([self.rev_vocab.get(t, self.unk_id) for t in utt], floor, id_feat))
                results.append(temp)
            return results
        id_train = _to_id_corpus(self.train_corpus[self.dialog_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.dialog_id])
        id_test = _to_id_corpus(self.test_corpus[self.dialog_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_meta_corpus(self):
        def _to_id_corpus(data):
            results = []
            for m_meta, o_meta, topic in data:
                results.append((m_meta, o_meta, self.rev_topic_vocab[topic]))
            return results

        id_train = _to_id_corpus(self.train_corpus[self.meta_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.meta_id])
        id_test = _to_id_corpus(self.test_corpus[self.meta_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}
