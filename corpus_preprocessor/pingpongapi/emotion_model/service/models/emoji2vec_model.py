import pickle
from scipy import spatial


class Emoji2VecModel:
    def __init__(self, emoji2vec_fname):
        self.emoji2vec_fname = emoji2vec_fname

        with open(emoji2vec_fname, 'rb') as file:
            self.vectors = pickle.load(file)

    def similarity(self, key1, key2):
        return 1 - spatial.distance.cosine(self.vectors[key1], self.vectors[key2])
