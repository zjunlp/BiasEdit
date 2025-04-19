# import sys
# import os
# sys.path.insert(0, os.path.abspath('..'))
# import json, string, random, logging, copy, random
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from utils import EditBatchSampler, dict_to, scr
# from tqdm import tqdm
# import copy
import pandas as pd
import difflib

from typing import Dict

from data.base import BaseDataset

import  json, string, random, logging, copy, random, torch
from torch.utils.data import Dataset
from tqdm import tqdm
import copy
import math

def _get_span(seq1, seq2, operation):
        """This function extract spans that are shared between two sequences."""
        seq1 = [str(x) for x in seq1.tolist()]
        seq2 = [str(x) for x in seq2.tolist()]

        matcher = difflib.SequenceMatcher(None, seq1, seq2)
        template1, template2 = [], []
        for op in matcher.get_opcodes():
            # each op is a list of tuple:
            # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
            # possible operation: replace, insert, equal
            # https://docs.python.org/3/library/difflib.html
            if (operation == "equal" and op[0] == "equal") or (
                operation == "diff" and op[0] != "equal"
            ):
                template1 += [x for x in range(op[1], op[2], 1)]
                template2 += [x for x in range(op[3], op[4], 1)]

        return template1, template2

class CrowsDataset(Dataset):
        # for d in data:
        #     self.data.append({k: d[k] for k in ["id", "target", "bias_type", "context", "data"]})

    def __init__(
        self,
        modelname,
        config,
        data_path,
        tokenizer,
        device,
        max_length=64
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self.data = []
        self.config = config
        self.device = device
        self.iscausal = "gpt" in modelname or "llama" in modelname or "mistral" in modelname or "gemma" in modelname
        if not self.iscausal:
            self.mask_token = tokenizer.mask_token
            self.mask_token_id = tokenizer.mask_token_id
        elif "gpt" in modelname or "llama3" in modelname or "mistral" in modelname:
            self._tokenizer.pad_token = tokenizer.eos_token
        elif "llama2" in modelname:
            self._tokenizer.pad_token = tokenizer.unk_token

        data = pd.read_csv(data_path)
        for index, row in data.iterrows():
            self.data.append({"anti": row["sent_less"], "stereo": row["sent_more"]})
        
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


    def collate_fn(self, batch):
        for b in batch:
            if "gpt" in self._tokenizer.__class__.__name__.lower():
                anti_sentences = self._tokenizer.bos_token + b['anti'] 
                stereo_sentences = self._tokenizer.bos_token + b['stereo']
            else:
                anti_sentences = b['anti']
                stereo_sentences = b['stereo']
            b['input_sentence'] = {"anti": anti_sentences, "stereo": stereo_sentences}
        
        edit = []

        if len(batch) < self.config.n_edits:
            for n_batch in range(math.ceil(len(batch) / self.config.batch_size)):
                if (n_batch + 1) * self.config.batch_size < len(batch):
                    batches = self.tok_samples(batch[n_batch * self.config.batch_size : (n_batch + 1) * self.config.batch_size])
                else:
                    batches = self.tok_samples(batch[n_batch * self.config.batch_size : ])
                edit.append(batches['edit'].to(self.device))
        else:
            for n_batch in range(math.ceil(self.config.n_edits / self.config.batch_size)):
                if (n_batch + 1) * self.config.batch_size < len(batch):
                    batches = self.tok_samples(batch[n_batch * self.config.batch_size : (n_batch + 1) * self.config.batch_size])
                else:
                    batches = self.tok_samples(batch[n_batch * self.config.batch_size : ])
                edit.append(batches['edit'].to(self.device))
        
        return {"edit": edit}

    def tok_samples(self, new_batch):
        batches = {}
        anti_srcs = [b["input_sentence"]["anti"] for b in new_batch]
        stereo_srcs = [b["input_sentence"]["stereo"] for b in new_batch]
        edit = anti_srcs + stereo_srcs

        inputs = self._tokenizer(
            edit,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        inputs['labels'] = copy.deepcopy(inputs['input_ids'])
        for batchin_idx in range(len(inputs['labels'])):
            inputs['labels'][batchin_idx] = torch.where(inputs['labels'][batchin_idx]!=self._tokenizer.pad_token_id, inputs['input_ids'][batchin_idx], -100)
            inputs['labels'][batchin_idx][0] = -100
        batches['edit'] = inputs
        return batches



