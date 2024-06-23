from typing import Union, Tuple, List, Dict
from omegaconf import DictConfig

import math
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import transformers


    

def make_loader(
    config: DictConfig,
    data_class
) -> Tuple[DataLoader]:
    
    tok = getattr(transformers, config.model.tok_name).from_pretrained(config.model.name_or_path)

    train_set = data_class(
        config.model.layers,
        config.data,
        config.data.train_path,
        tok,
        config.model_device
    )

    valid_set = data_class(
        config.model.layers,
        config.data,
        config.data.valid_path,
        tok,
        config.model_device
    )

    train_loader = DataLoader(
        train_set,
        config.data.n_edits,
        # shuffle=True,
        collate_fn = train_set.collate_fn,
        drop_last = True
    )

    valid_loader = DataLoader(
        valid_set,
        config.data.n_edits,
        # shuffle=True,
        collate_fn = valid_set.collate_fn,
    )

    return train_loader, valid_loader