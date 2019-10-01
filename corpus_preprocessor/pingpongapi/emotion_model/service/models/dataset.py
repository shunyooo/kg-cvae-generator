import numpy as np


class Vectorizer:
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3
    pre_vocab = ['PAD', 'UNK', 'SOS', 'EOS']

    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.idx_to_vocab = self.pre_vocab \
                                + [line.strip().split('\t')[0] for line in f]
            self.vocab_to_idx = {v: i for i, v in enumerate(self.idx_to_vocab)}

    @property
    def num_vocabs(self):
        return len(self.idx_to_vocab)

    def vectorize(self, sentence):
        return [self.vocab_to_idx.get(token, self.UNK) for token in
                sentence.split()]

    def stringize(self, tokens, delimiter=' '):
        return delimiter.join([self.idx_to_vocab[idx] for idx in tokens])


class Dataset:
    def __init__(self,
                 vocab_fname,
                 idx_to_class,
                 feed_tensors,
                 add_EOS=True):
        self.vectorizer = Vectorizer(vocab_fname)
        self.idx_to_class = idx_to_class
        self.class_to_idx = {cls: i for i, cls in enumerate(idx_to_class)}
        self.num_class = len(idx_to_class)
        self.add_eos = add_EOS
        self.feed_tensors = feed_tensors

    def infer_batch_feed_dict(self, sentences):
        instances = []
        for sentence in sentences:
            if type(sentence) == np.ndarray:
                sentence = sentence[0]
            tokens = self.vectorizer.vectorize(sentence)
            if self.add_eos:
                tokens += [self.vectorizer.EOS]
            instances.append({'tokens': tokens})

        return self._batchify(instances)

    def _batchify(self, instances):
        lengths = np.array([len(inst['tokens']) for inst in instances],
                           dtype=np.int32)
        max_lengths = max(lengths)

        pad_idx = self.vectorizer.PAD
        if pad_idx == 0:
            tokens = np.zeros((len(instances), max_lengths), dtype=np.int32)
        else:
            tokens = np.ones((len(instances), max_lengths), dtype=np.int32) \
                     * pad_idx

        for i, inst in enumerate(instances):
            token_arr = inst['tokens']
            tokens[i, :len(token_arr)] += token_arr

        batch_dict = {
            self.feed_tensors['tokens']: tokens,
            self.feed_tensors['lengths']: lengths,
        }

        return batch_dict