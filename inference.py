import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import utils
import torch
from torch.utils.data import DataLoader

from data_apis.corpus import PingpongDialogCorpus
from data_apis.dataset import CVAEDataset
from data_apis.dataloader import get_cvae_collate

from model.cvae import CVAEModel

import tqdm

corpus_config_path = './config/korean/cvae_corpus_small.json'
dataset_config_path = './config/korean/cvae_dataset.json'
trainer_config_path = './config/korean/cvae_trainer_small.json'
model_config_path = './config/korean/cvae_model.json'

overall = {
    "work_dir": "./work",
    "log_dir": "log",
    "model_dir": "weights",
    "test_dir": "test"
}


def inference(model, data_loader, trainer_config):
    test_outputs = []

    iterator = tqdm.tqdm(data_loader, desc="Inference")

    is_test_multi_da = trainer_config["is_test_multi_da"]
    num_samples = trainer_config["num_samples"]

    for model_input in iterator:
        model_input["is_train"] = False
        model_input["is_test_multi_da"] = is_test_multi_da
        model_input["num_samples"] = num_samples

        with torch.no_grad():
            model_output = model.forward(model_input)
        test_output = {"model_input": model_input, "model_output": model_output}

        test_outputs.append(test_output)

    topics = ["LOVER", "FRIEND"]
    profiles = ["STUDENT", "COLLEGIAN", "CIVILIAN", "FEMALE", "MALE"]
    prof_len = len(profiles)

    final = []

    for output in test_outputs:
        segment_indices = []
        context_lens = output["model_output"]["context_lens"].tolist()
        for i in range(1, len(context_lens)):
            if context_lens[i] <= context_lens[i - 1]:
                segment_indices.append(i - 1)
        segment_indices.append(len(context_lens) - 1)

        for i in segment_indices:
            out_dict = {"relation": topics[output["model_input"]["topics"][i]],
                        "my_profile": [profiles[j] for j in range(prof_len) if
                                       output["model_input"]["my_profile"][i][j]],
                        "ot_profile": [profiles[j] for j in range(prof_len) if
                                       output["model_input"]["ot_profile"][i][j]],
                        "gt_sentiment": output["model_output"]["real_output_das"][i],
                        "contexts": output["model_output"]["context_sents"][i],
                        "generated": output["model_output"]["output_sents"][i],
                        "samples": [output["model_output"]["sampled_output_sents"][j][i] for j in
                                    range(num_samples - 1)]}

            if is_test_multi_da:
                for da in model.da_vocab:
                    out_dict[da] = output["model_output"]["ctrl_output_sents"][da][i]

            out_dict["ground_truth"] = output["model_output"]["real_output_sents"][i]
            out_dict["predicted_sentiment"] = output["model_output"]["output_das"][i]

            final.append(out_dict)

    return final


def main():
    corpus_config = utils.load_config(corpus_config_path)
    corpus = PingpongDialogCorpus(corpus_config)

    dial_corpus = corpus.get_dialog_corpus()
    meta_corpus = corpus.get_meta_corpus()

    test_meta = meta_corpus.get("test")
    test_dial = dial_corpus.get("test")

    dataset_config = utils.load_config(dataset_config_path)
    utt_per_case = dataset_config["utt_per_case"]
    max_utt_size = dataset_config["max_utt_len"]

    test_set = CVAEDataset("Test", test_dial, test_meta, "kor", dataset_config)

    cvae_collate = get_cvae_collate(utt_per_case, max_utt_size)

    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, collate_fn=cvae_collate)

    trainer_config = utils.load_config(trainer_config_path)
    model_config = utils.load_config(model_config_path)

    log_dir_path = os.path.join(overall["work_dir"], overall["log_dir"])
    model_dir_path = os.path.join(overall["work_dir"], overall["model_dir"])
    test_dir_path = os.path.join(overall["work_dir"], overall["test_dir"])
    dir_paths = [log_dir_path, model_dir_path, test_dir_path]
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    model = CVAEModel(dataset_config, model_config, corpus)
    model.load_state_dict(torch.load(trainer_config["model_path"].format(trainer_config["epoch"] - trainer_config["save_epoch_step"])))
    model.eval()
    model.cuda()

    json = inference(model, test_loader, trainer_config)


if __name__ == "__main__":
    main()