import numpy as np
from xgboost import XGBRegressor, Booster


class RescoreEmojiModel:
    def __init__(self, emotion_model, calibrated_model, emojiness_model, xgb_fname):
        self.emotion_model = emotion_model
        self.calibrated_model = calibrated_model
        self.emojiness_model = emojiness_model

        self.xgb = XGBRegressor(tree_method='gpu_hist')
        booster = Booster()
        booster.load_model(xgb_fname)
        self.xgb._Booster = booster

        self.class_to_idx = {name: i for i, name in enumerate(self.emotion_model.idx_to_class)}
        self.class_to_idx.update({name: i for i, name in enumerate(self.emotion_model.idx_to_class_name)})

    def feature_extraction(self, query):
        result_features = []

        # emoji index
        emoji_onehot_vec = np.zeros(45)

        result_features.extend(emoji_onehot_vec)

        # emoji prob
        emoji_class_probs, posneg = self.emotion_model.predict_proba(query)

        # posneg
        result_features.append(float(posneg[0]))

        # all emojies' probs
        result_features.extend(emoji_class_probs[0])

        # emojiness
        emojiness = self.emojiness_model.predict_proba(query)[0][0]
        result_features.append(emojiness)

        return result_features, emoji_class_probs, posneg

    def predict(self, sentence, class_name=False, topn=5):
        default_feature, probs, posneg = self.feature_extraction(sentence)
        topn_emoji = sorted(list(enumerate(self.calibrated_model.calibration_proba(probs)[0])),
                            key=lambda x: x[1], reverse=True)[:max(20, topn)]
        topn_emoji = [i for i, prob in topn_emoji]

        features = np.tile(np.array(default_feature), (len(topn_emoji), 1))
        for i, idx in enumerate(topn_emoji):
            features[i, idx] = 1

        scores = self.xgb.predict(features)

        if class_name is True:
            result = sorted(
                [(float(score), self.emotion_model.idx_to_class_name[i]) for i, score in zip(topn_emoji, scores)],
                reverse=True)
        else:
            result = sorted(
                [(float(score), self.emotion_model.idx_to_class[i]) for i, score in zip(topn_emoji, scores)],
                reverse=True)

        return result[:topn], float(posneg)
