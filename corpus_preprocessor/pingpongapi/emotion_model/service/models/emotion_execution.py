from scatternlp.preprocessor.helper import PreprocessorHelper
from .emoji_model import EmojiModel
from .calibrated_emoji_model import CalibratedEmojiModel
from .emojiness_model import EmojinessModel
from .rescore_emoji_model import RescoreEmojiModel
from .sentiment_tagging_model import SentimentTaggingModel
import numpy as np


class EmotionExecution:
    def __init__(self, config):
        self.config = config
        preprocessor_config = {
            "space_model_path": config["preprocess"]["space_model"],
            "cohesion_path": config["preprocess"]["cohesion"],
            "typo_correction_path": config["preprocess"]["typo_correction"],
            "system_msg": config["preprocess"]["system_msg"]
        }
        self.preprocessor_helper = PreprocessorHelper(**preprocessor_config)
        self.preprocess_pipeline = self.preprocessor_helper._get_tokenize_pipeline()

        self.emoji_model = EmojiModel(config['emoji']['vocab'], config['emoji']['emoji_model'])
        self.emoji_model.load_graph()
        self.emoji_model.init_dataset()

        self.calibrated_emoji_model = CalibratedEmojiModel(self.emoji_model, config['emoji']['calibrated_classifier'])

        self.emojiness_model = EmojinessModel(config['emojiness']['vocab'], config['emojiness']['emojiness_model'])
        self.emojiness_model.load_graph()
        self.emojiness_model.init_dataset()

        self.rescore_model = RescoreEmojiModel(
            self.emoji_model,
            self.calibrated_emoji_model,
            self.emojiness_model,
            config['emoji']['ranking_model']
        )

        self.sentiment_model = SentimentTaggingModel(self.emoji_model, self.emojiness_model,
                                                     config['sentiment']['ranking_model'])

        # test
        #self.process('초기화 및 테스트용 쿼리')
    def batch_pos_neg_process(self, queries):
        queries = [self.preprocess_pipeline.run(query) for query in queries]

        _, posnegs = self.emoji_model.predict_proba(queries)
        return [float(posneg[0]) for posneg in posnegs]

    def pos_neg_process(self, query):
        queries = [self.preprocess_pipeline.run(query)]

        _, posnegs = self.emoji_model.predict_proba(queries)
        return float(posnegs[0][0])

    def batch_process(self, queries, emotion=True, topn=5, sentiment=True):
        queries = [self.preprocess_pipeline.run(query) for query in queries]



        return self.sentiment_model.batch_predict(queries)

    def process_one(self, query):
        query = self.preprocess_pipeline.run(query)
        return self.sentiment_model.predict(query)

    def process(self, query, emotion=True, topn=5, sentiment=True):
        query = self.preprocess_pipeline.run(query)

        result = {}
        if emotion:
            result['emoji'], result['posneg'] = self.rescore_model.predict(query, topn=topn)
        if sentiment:
            #print(np.array(self.sentiment_model.feature_extraction(query)).shape)
            result['sentiment'] = self.sentiment_model.predict(query)

        return result
