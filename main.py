import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import utils
from torch.utils.data import DataLoader

from data_apis.corpus import PingpongDialogCorpus, SWDADialogCorpus
from data_apis.dataset import CVAEDataset
from data_apis.dataloader import get_cvae_collate

from trainer.cvae.trainer import CVAETrainer

from model.cvae import CVAEModel
import torch.autograd

eng_corpus_config_path = './config/english/cvae_corpus_eng.json'
eng_dataset_config_path = './config/english/cvae_dataset.json'
eng_trainer_config_path = './config/english/cvae_trainer.json'
eng_model_config_path = './config/english/cvae_model.json'


kor_corpus_config_path = './config/korean/cvae_corpus_small.json'
kor_dataset_config_path = './config/korean/cvae_dataset.json'
kor_trainer_config_path = './config/korean/cvae_trainer_small.json'
kor_model_config_path = './config/korean/cvae_model.json'


overall = {
  "work_dir": "./work",
  "log_dir": "log",
  "model_dir": "weights",
  "test_dir": "test"
}

language = "kor"

torch.autograd.set_detect_anomaly(True)

if language == "eng":
    corpus_config_path = eng_corpus_config_path
    dataset_config_path = eng_dataset_config_path
    trainer_config_path = eng_trainer_config_path
    model_config_path = eng_model_config_path
elif language == "kor":
    corpus_config_path = kor_corpus_config_path
    dataset_config_path = kor_dataset_config_path
    trainer_config_path = kor_trainer_config_path
    model_config_path = kor_model_config_path
else:
    corpus_config_path = kor_corpus_config_path
    dataset_config_path = kor_dataset_config_path
    trainer_config_path = kor_trainer_config_path
    model_config_path = kor_model_config_path


def main():
    # Generate Corpus
    corpus_config = utils.load_config(corpus_config_path)
    if language == "kor":
        corpus = PingpongDialogCorpus(corpus_config)
    else:
        corpus = SWDADialogCorpus(corpus_config)

    rev_topic_vocab = corpus.rev_topic_vocab
    print("rev_topic_vocab", rev_topic_vocab)

    rev_da_vocab = corpus.rev_dialog_act_vocab
    index_to_sentiment = {rev_da_vocab[key]: key for key in rev_da_vocab}

    dial_corpus = corpus.get_dialog_corpus()
    meta_corpus = corpus.get_meta_corpus()

    train_meta, valid_meta, test_meta = meta_corpus.get("train"), meta_corpus.get("valid"), meta_corpus.get("test")
    train_dial, valid_dial, test_dial = dial_corpus.get("train"), dial_corpus.get("valid"), dial_corpus.get("test")

    # Generate Dataset
    dataset_config = utils.load_config(dataset_config_path)
    utt_per_case = dataset_config["utt_per_case"]
    max_utt_size = dataset_config["max_utt_len"]

    train_set = CVAEDataset("Train", train_dial, train_meta, language, dataset_config)
    valid_set = CVAEDataset("Valid", valid_dial, valid_meta, language, dataset_config)
    test_set = CVAEDataset("Test", test_dial, test_meta, language, dataset_config)

    cvae_collate = get_cvae_collate(utt_per_case, max_utt_size)

    train_loader = DataLoader(train_set, batch_size=100, shuffle=True, collate_fn=cvae_collate)
    valid_loader = DataLoader(valid_set, batch_size=100, shuffle=False, collate_fn=cvae_collate)
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

    target_model = CVAEModel(dataset_config, model_config, corpus)
    target_model.cuda()
    cvae_trainer = CVAETrainer(trainer_config, target_model)

    output_reports = cvae_trainer.experiment(train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    main()
