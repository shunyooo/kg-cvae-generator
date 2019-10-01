import re
from konlpy.tag import Mecab


class Normalizer:
    def normalize(self, sent):
        pass


class SimpleNormalizer(Normalizer):
    def normalize(self, sent):
        sent = re.sub(r'<EMO>', ' ', sent)
        sent = re.sub(r'<URL>', ' ', sent)
        sent = re.sub(r'!+', '!', sent)
        sent = re.sub(r'\?+', '?', sent)
        sent = re.sub(r'~+', '~', sent)
        sent = re.sub(r'♥+', '♥', sent)
        sent = re.sub(r' {2,}', ' ', sent)
        sent = re.sub(r',+', ',', sent)
        sent = re.sub(r';{2,}', ';;', sent)
        sent = re.sub(r'\.{2,}', '...', sent)
        sent = re.sub(r' {2,}', ' ', sent)
        return sent


class Tokenizer:
    def tokenize(self, sent):
        pass


class MecabTokenizer(Tokenizer):
    def __init__(self):
        self.model = Mecab()

    def tokenize(self, sent):
        return self.model.morphs(sent)