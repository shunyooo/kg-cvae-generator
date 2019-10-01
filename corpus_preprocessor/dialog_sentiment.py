import json
import copy

import pingpongapi.emotion_model.utils as utils
import pingpongapi.emotion_model.service.models.emotion_execution as emotion_execution
import tqdm


class SentimentExtractor(object):
    def __init__(self, config_path="config_service.yaml"):
        config = utils.load_config(config_path)
        self.execution_model = emotion_execution.EmotionExecution(config)

    def extract_sentiment_batch(self, sent_batch):
        return self.execution_model.batch_pos_neg_process(sent_batch)

    def extract_sentiment_from_json(self, input_json_path, output_json_path, max_batch_size):
        with open(input_json_path, "r") as json_reader:
            json_content = json.load(json_reader)
        output_json_content = copy.deepcopy(json_content)

        input_batch = []
        input_idx_batch = []
        overall_output = []
        for session_idx, session in enumerate(tqdm.tqdm(json_content)):
            session_utts = session['utts']
            for turn_idx, turn_info in enumerate(session_utts):
                real_sent = turn_info[2]
                input_idx_batch.append((session_idx, turn_idx))
                input_batch.append(turn_info[2])
            if len(input_batch) > max_batch_size:
                output_batch = self.extract_sentiment_batch(input_batch)
                for idx, sentiment in enumerate(output_batch):
                    target_session_idx, target_turn_idx = input_idx_batch[idx]
                    target_line = output_json_content[target_session_idx]['utts'][target_turn_idx]
                    target_line.append(sentiment)
                input_batch = []
                input_idx_batch = []
                overall_output.extend(output_batch)
        if len(input_batch) != 0:
            output_batch = self.extract_sentiment_batch(input_batch)
            for idx, sentiment in enumerate(output_batch):
                target_session_idx, target_turn_idx = input_idx_batch[idx]
                target_line = output_json_content[target_session_idx]['utts'][target_turn_idx]
                target_line.append(sentiment)
            input_batch = []
            input_idx_batch = []
            overall_output.extend(output_batch)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json_content, f, ensure_ascii=False, indent=4)
        return overall_output



