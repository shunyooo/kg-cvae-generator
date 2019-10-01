import tensorflow as tf

from .utils import load_config, load_main_config, get_session
from .dataset import Dataset


class EmojinessModel:
    def __init__(self, vocab_fname=None, emojiness_fname=None):
        self.vocab_fname = vocab_fname
        self.model_fname = emojiness_fname
        self.idx_to_class = ['emojiness', 'nonemojiness']
        self.session = None
        self.saver = None
        self.graph = None
        self.inference_feed_dict = None
        self.outputs = None
        self.dataset = None

    def load_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            if not self.session:
                self.session = get_session()

            self.saver = tf.train.import_meta_graph(self.model_fname + '.meta')
            self.saver.restore(self.session, self.model_fname)

        self.inference_feed_dict = self.get_infer_feed_dict()
        self.outputs = self.get_outputs()

    def get_infer_feed_dict(self):
        tokens = self.graph.get_tensor_by_name('tokens:0')
        lengths = self.graph.get_tensor_by_name('lengths:0')

        return {'tokens': tokens, 'lengths': lengths}

    def get_outputs(self):
        predictions = self.graph.get_tensor_by_name('Softmax:0')

        return [predictions]

    def init_dataset(self):
        if not self.inference_feed_dict:
            self.inference_feed_dict = self.get_infer_feed_dict()

        self.dataset = Dataset(self.vocab_fname, self.idx_to_class, self.inference_feed_dict)

    def predict(self, sentences):
        results = self.predict_proba(sentences)

        return [result[0] for result in results]

    def predict_proba(self, sentences):
        if type(sentences) == str:
            sentences = [sentences]

        fd = self.dataset.infer_batch_feed_dict(sentences)
        results = self.session.run(self.outputs, fd)

        return results[0]


