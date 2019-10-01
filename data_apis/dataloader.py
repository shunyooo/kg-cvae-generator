import torch
from torch.utils.data.dataloader import default_collate
import numpy as np


def get_cvae_collate(utt_per_case, max_utt_size):
    def cvae_collate(cvae_data_list):
        batch_size = len(cvae_data_list)
        collate_result = default_collate(cvae_data_list)
        max_vec_context_lens = max(collate_result["context_lens"]).item()
        max_out_lens = max(collate_result["out_lens"]).item()

        vec_context = np.zeros((batch_size, utt_per_case, max_utt_size))
        vec_floors = np.zeros((batch_size, utt_per_case))
        vec_outs = np.zeros((batch_size, max_utt_size))

        for idx, item in enumerate(cvae_data_list):
            context_idx = item["context_lens"].item()
            outlen_idx = item["out_lens"].item()

            vec_context[idx] = item["context_utts"]
            vec_floors[idx] = item["floors"]
            vec_outs[idx] = item["out_utts"]

        collate_result["vec_context"] = torch.LongTensor(vec_context)
        collate_result["vec_floors"] = torch.LongTensor(vec_floors)
        collate_result["vec_outs"] = torch.LongTensor(vec_outs)
        return collate_result
    return cvae_collate
