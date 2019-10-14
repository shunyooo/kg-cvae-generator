"""
Copyright 2019 Pingpong AI Research, ScatterLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import os
import utils
from torch.utils.data import DataLoader

from data_apis.corpus_kor import PingpongDialogCorpus
from data_apis.dataset import CVAEDataset
from data_apis.dataloader import get_cvae_collate

from trainer.cvae.trainer import CVAETrainer

from model.cvae import CVAEModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

corpus_config_path = './config/korean/cvae_corpus_kor.json'
dataset_config_path = './config/korean/cvae_dataset_kor.json'
trainer_config_path = './config/korean/cvae_trainer_kor.json'
model_config_path = './config/korean/cvae_model_kor.json'


overall = {
  "work_dir": "./work",
  "log_dir": "log",
  "model_dir": "weights",
  "test_dir": "test"
}

language = "kor"


def main():
    # Generate Corpus
    corpus_config = utils.load_config(corpus_config_path)
    corpus = PingpongDialogCorpus(corpus_config)

    dial_corpus = corpus.get_dialog_corpus()
    meta_corpus = corpus.get_meta_corpus()

    train_meta, valid_meta, test_meta = \
        meta_corpus.get("train"), meta_corpus.get("valid"), meta_corpus.get("test")
    train_dial, valid_dial, test_dial = \
        dial_corpus.get("train"), dial_corpus.get("valid"), dial_corpus.get("test")

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

    target_model = CVAEModel(dataset_config, model_config, corpus)
    target_model.cuda()
    cvae_trainer = CVAETrainer(trainer_config, target_model)

    output_reports = cvae_trainer.experiment(train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    main()
