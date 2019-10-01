import numpy as np
import pickle


class CalibratedEmojiModel:
    def __init__(self, emoji_model, calibrated_classifier_fname=None):
        self.emoji_model = emoji_model
        self.calibrated_classifier_fname = calibrated_classifier_fname

        with open(self.calibrated_classifier_fname, 'rb') as file:
            calib_model = pickle.load(file=file)

        calib_model.base_estimator = emoji_model
        calib_model.calibrated_classifiers_[0].base_estimator = emoji_model

        self.calib_model = calib_model

    def calibration_proba(self, probs):
        n_classes = len(self.emoji_model.classes_)

        idx_pos_class = np.arange(n_classes)

        for k, this_df, calibrator in \
                zip(idx_pos_class, probs.T, self.calib_model.calibrated_classifiers_[0].calibrators_):
            if n_classes == 2:
                k += 1
            probs[:, k] = calibrator.predict(this_df)

        # Normalize the probabilities
        if n_classes == 2:
            probs[:, 0] = 1. - probs[:, 1]
        else:
            probs /= np.sum(probs, axis=1)[:, np.newaxis]

        # XXX : for some reason all probas can be 0
        probs[np.isnan(probs)] = 1. / n_classes

        # Deal with cases where the predicted probability minimally exceeds 1.0
        probs[(1.0 < probs) & (probs <= 1.0 + 1e-5)] = 1.0

        return probs

    def predict(self, sentences, class_name=False, topn=3):
        emoji_results, posneg_results = self.predict_proba(sentences)

        temp_emoji_results = []
        temp_posneg_results = []

        for emj_res, posneg_res in zip(emoji_results, posneg_results):
            emj_res = sorted(self.emoji_model.sorted_emojies(emj_res, class_name), reverse=True)[:topn]
            posneg_res = float(posneg_res)

            temp_emoji_results.append(emj_res)
            temp_posneg_results.append(posneg_res)

        return temp_emoji_results, temp_posneg_results

    def predict_proba(self, sentences):
        emoji_results, posneg_results = self.emoji_model.predict_proba(sentences)
        emoji_results = self.calibration_proba(emoji_results)

        return emoji_results, posneg_results