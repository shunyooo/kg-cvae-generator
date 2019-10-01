import tensorflow as tf

from .utils import load_config, load_main_config, get_session
from .dataset import Dataset


class EmojiModel:
    def __init__(self,
                 vocab_fname=None,
                 model_fname=None,
                 idx_to_class=None,
                 idx_to_class_name=None):

        if not idx_to_class:
            idx_to_class = [
                'emoji13', 'emoji26', 'emoji28', 'emoji38', 'emoji33',
                'emoji15', 'emoji09', 'emoji14', 'emoji20', 'emoji37',
                'emoji43', 'emoji18', 'emoji39', 'emoji05', 'emoji04',
                'emoji25', 'emoji32', 'emoji42', 'emoji44', 'emoji03',
                'emoji12', 'emoji23', 'emoji16', 'emoji02', 'emoji27',
                'emoji21', 'emoji17', 'emoji36', 'emoji11', 'emoji40',
                'emoji19', 'emoji45', 'emoji22', 'emoji10', 'emoji30',
                'emoji07', 'emoji35', 'emoji34', 'emoji41', 'emoji06',
                'emoji01', 'emoji08', 'emoji31', 'emoji24', 'emoji29'
            ]
        
        if not idx_to_class_name:
            idx_to_class_name = [
                '감동', '곤란', '괴로움', '굿나잇', '궁금/고민', 
                '근심', '긍정적인 부끄러움', '꺄아 신남', '끄아악 공포', '노노',
                '노래', '놀람', '메롱', '무표정', '미소',
                '민망', '반짝', '부탁', '비밀', '뽀뽀',
                '뿌듯함, 자랑스러움, 우쭐댐', '삐짐', '슬픈 눈물', '실연', '아픔',
                '약간 부정적', '약간 장난스러운 눈물', '오케이', '웃픔 혹은 폭소', '윙크',
                '이럴수가', '잘난척', '장난스러운 심각함', '장난스러운 웃음', '졸림',
                '진심이 담긴 미소', '최고', '최악', '축하', '크게 웃기',
                '하트', '행복한 크게 웃기', '헤롱', '화남', '힘듦과 당황'
            ]

        self.vocab_fname = vocab_fname
        self.model_fname = model_fname
        self.idx_to_class = idx_to_class
        self.idx_to_class_name = idx_to_class_name
        self.session = None
        self.saver = None
        self.graph = None
        self.inference_feed_dict = None
        self.outputs = None
        self.dataset = None
        self.classes_ = idx_to_class_name

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
        posneg_preds = self.graph.get_tensor_by_name('Tanh:0')

        return [predictions, posneg_preds]

    def init_dataset(self):
        if not self.inference_feed_dict:
            self.inference_feed_dict = self.get_infer_feed_dict()

        self.dataset = Dataset(self.vocab_fname, self.idx_to_class, self.inference_feed_dict)

    def sorted_emojies(self, probs, class_name=False):
        if class_name is False:
            result = [(float(score), emoji) for emoji, score in zip(self.idx_to_class, probs)]
        else:
            result = [(float(score), emoji) for emoji, score in zip(self.idx_to_class_name, probs)]

        return sorted(result, reverse=True)

    def predict(self, sentences, class_name=False, topn=3):
        emoji_results, posneg_results = self.predict_proba(sentences)
    
        temp_emoji_results = []
        temp_posneg_results = []
        
        for emj_res, posneg_res in zip(emoji_results, posneg_results):
            emj_res = sorted(self.sorted_emojies(emj_res, class_name), reverse=True)[:topn]
            posneg_res = float(posneg_res)
            
            temp_emoji_results.append(emj_res)
            temp_posneg_results.append(posneg_res)

        return temp_emoji_results, temp_posneg_results
    
    def predict_proba(self, sentences):
        if type(sentences) == str:
            sentences = [sentences]

        fd = self.dataset.infer_batch_feed_dict(sentences)
        emoji_results, posneg_results = self.session.run(self.outputs, fd)
        
        return emoji_results, posneg_results