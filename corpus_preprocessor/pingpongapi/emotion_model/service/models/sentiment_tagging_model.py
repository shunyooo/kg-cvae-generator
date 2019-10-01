import numpy as np
from xgboost import XGBRegressor, Booster
import tqdm


class SentimentTaggingModel:
    def __init__(self, emoji_model, emojiness_model, xgb_fname):
        self.idx_to_sentiment_name = [
            '연락해', '굿모닝', '굿밤', '안녕', '아니', '싫어', '그래', '좋아', '희망사항', '궁금', '체념',
            '한숨', '심심함', '귀찮음', '깜짝', '무서움', '화이팅', '결심', '스트레스', '짜증', '힘듦',
            '화남', '흥!', '피곤', '아쉬움', '걱정', '불안', '슬픔', '아픔', '부러움', '설렘', '행복',
            '신남', '킥킥', '최고', '사랑', '미안', '고마움'
        ]

        self.emoji_model = emoji_model
        self.emojiness_model = emojiness_model

        self.xgb = XGBRegressor(tree_method='gpu_hist')
        booster = Booster()
        booster.load_model(xgb_fname)
        self.xgb._Booster = booster

    def feature_extraction(self, query):
        result_features = []

        # emoji index
        emoji_onehot_vec = np.zeros(len(self.idx_to_sentiment_name))

        result_features.extend(emoji_onehot_vec)

        # emoji prob
        emoji_class_probs, posneg = self.emoji_model.predict_proba(query)

        # posneg
        result_features.append(float(posneg[0]))

        # all emojies' probs
        result_features.extend(emoji_class_probs[0])

        # emojiness
        emojiness = self.emojiness_model.predict_proba(query)[0][0]
        result_features.append(emojiness)

        return result_features

    def feature_extraction_batch(self, queries):
        emoji_size = len(self.idx_to_sentiment_name)
        result_feature_batch = [[0.0]*(emoji_size)]*len(queries)

        emoji_class_probs, posnegs = self.emoji_model.predict_proba(queries)
        emojiness = self.emojiness_model.predict(queries)

        for idx, posneg in enumerate(posnegs):
            result_feature_batch[idx].append(float(posneg[0]))
            result_feature_batch[idx].append(emojiness[idx])

        # numpy.array(result_feature)
        return result_feature_batch

    def predict(self, sentence):
        sentiment_features = self.feature_extraction(sentence)
        sentiment_features = np.tile(np.array(sentiment_features), (len(self.idx_to_sentiment_name), 1))
        for i in range(len(self.idx_to_sentiment_name)):
            sentiment_features[i, i] = 1

        scores = self.xgb.predict(sentiment_features)
        return sorted([(key, float(value)) for key, value in zip(self.idx_to_sentiment_name, scores)],
                      key=lambda x: x[1], reverse=True)

    def batch_predict(self, sentence_batches):
        group_len = len(self.idx_to_sentiment_name)
        sentence_batch_len = len(sentence_batches)
        overall_sentiment_features = []

        feature_extracted_batch = self.feature_extraction_batch(sentence_batches)

        for sentence_idx in tqdm.tqdm(range(sentence_batch_len)):
            sentiment_features = [feature_extracted_batch[sentence_idx]] * group_len
            overall_sentiment_features.extend(sentiment_features)
        overall_sentiment_features = np.array(overall_sentiment_features)
        print(overall_sentiment_features.shape)
        for i in tqdm.tqdm(range(group_len)):
            for j in range(len(sentence_batches)):
                overall_sentiment_features[i+group_len*j, i] = 1

        print("inference started")
        overall_scores = self.xgb.predict(overall_sentiment_features)
        print("inference ended")
        overall_scores_list = [overall_scores[i*group_len:(i+1)*group_len] for i in range(len(sentence_batches))]
        result_list = []
        for scores in overall_scores_list:
            #print(scores)
            #print(scores.shape)
            result_scores = sorted([(key, float(value)) for key, value in zip(self.idx_to_sentiment_name, scores)], key=lambda x: x[1], reverse=True)
            result_list.append(result_scores)

        return result_list
