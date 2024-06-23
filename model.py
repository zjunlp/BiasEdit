from omegaconf import DictConfig

import torch
import torch.nn as nn

import transformers
from transformers import AutoModel

from util import get_module


def make_model(config: DictConfig):
    
    model_class = getattr(transformers, config.class_name)
    model = model_class.from_pretrained(config.name_or_path)

    if config.half:
        model.bfloat16()

    for param in model.parameters():
        param.requires_grad = False
        
    for module_name in config.edit_modules:
        module = get_module(model, module_name)
        module.weight.requires_grad = True
        
    return model