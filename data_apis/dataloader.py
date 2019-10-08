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


import torch
from torch.utils.data.dataloader import default_collate
import numpy as np


def get_cvae_collate(utt_per_case, max_utt_size):
    def cvae_collate(cvae_data_list):
        """
        collate_fn for CVAE dataloader.
        collate function collates lists of samples into batches
        cvae_collate sets maximum lengths of input contexts and ground truth response,
        and also tensorizes preceding utterances, floors, and ground truth response
        :param cvae_data_list: list of data instances
        :return: collated batch
        """
        batch_size = len(cvae_data_list)
        collate_result = default_collate(cvae_data_list)

        vec_context = np.zeros((batch_size, utt_per_case, max_utt_size))
        vec_floors = np.zeros((batch_size, utt_per_case))
        vec_outs = np.zeros((batch_size, max_utt_size))

        for idx, item in enumerate(cvae_data_list):
            vec_context[idx] = item["context_utts"]
            vec_floors[idx] = item["floors"]
            vec_outs[idx] = item["out_utts"]

        collate_result["vec_context"] = torch.LongTensor(vec_context)
        collate_result["vec_floors"] = torch.LongTensor(vec_floors)
        collate_result["vec_outs"] = torch.LongTensor(vec_outs)
        return collate_result
    return cvae_collate
