from typing import Union, Tuple, List, Dict
from omegaconf import DictConfig
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.pytorch_utils import Conv1D

def empty_cache(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        suffixes = os.listdir(path)
        for s in suffixes:
            os.remove(os.path.join(path, s))

def get_module(module: nn.Module, module_name: str) -> nn.Module:
    
    for name in module_name.split("."):
        module = getattr(module, name)
    return module

def get_shape(module: Union[nn.Linear, Conv1D]) -> Tuple[int]:
    
    shape = tuple(module.weight.shape)
    return shape[::-1] if isinstance(module, nn.Linear) else shape
    
def cross_entropy(
    logits: torch.FloatTensor,
    labels: torch.LongTensor
):
    if len(logits.shape) == 2:

        return F.binary_cross_entropy_with_logits(logits, labels)

    if len(logits.shape) == 3:

        ans_indice = torch.where(labels != -100)
        
        logits = logits[ans_indice]
        labels = labels[ans_indice]
        
        return F.cross_entropy(logits, labels)




def log(x: torch.FloatTensor) -> torch.FloatTensor:
    return (x + torch.finfo(x.dtype).eps).log()

def kl_div(
    refer_logits: torch.FloatTensor,
    logits: torch.FloatTensor,
    labels: torch.LongTensor
) -> torch.Tensor:
    
    if len(logits.shape) == 2:

        refer_probs = F.sigmoid(refer_logits)
        probs = F.sigmoid(logits)

        return (refer_probs * (log(refer_probs) - log(probs))) + ((1 - refer_probs) * (log(1 - refer_probs) - log(1 - probs)))
    
    if len(logits.shape) == 3:

        ans_indice = torch.where(labels != -100)
        
        refer_logits = refer_logits[ans_indice]
        logits = logits[ans_indice]
        
        refer_log_probs = refer_logits.log_softmax(-1)
        log_probs = logits.log_softmax(-1)
        
        return F.kl_div(
            log_probs,
            refer_log_probs,
            reduction = "batchmean",
            log_target = True
        )
    
def succ_ratios(
    logits: torch.FloatTensor,
    labels: torch.LongTensor
) -> List[float]:
    
    if len(logits.shape) == 2:

        return ((logits > 0) == labels).squeeze(-1).to("cpu").numpy().tolist()
    
    if len(logits.shape) == 3:

        n_corr = (logits.argmax(-1) == labels).sum(-1)
        n_tokens = (labels != -100).sum(-1)
        
        return (n_corr / n_tokens).to("cpu").numpy().tolist()


class Tracer:

    def __init__(
        self,
        module: nn.Module,
        cache_mask: torch.LongTensor
    ):
        cache_indices = torch.where(cache_mask) # Answer's True positions

        def forward_hook(
            module: nn.Module,
            inputs: Tuple[torch.FloatTensor],
            outputs: Tuple[torch.FloatTensor]
        ):
            self.keys = inputs[0][cache_indices].detach()
            
        def backward_hook(
            module: nn.Module,
            inputs_grad: Tuple[torch.FloatTensor],
            outputs_grad: Tuple[torch.FloatTensor]
        ):
            self.values_grad = outputs_grad[0][cache_indices].detach()

        self.handles = [
            module.register_forward_hook(forward_hook),
            module.register_full_backward_hook(backward_hook)
        ]


class TracerDict(dict):
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        tuples: Dict[str, torch.LongTensor]
    ):
        
        if any("encoder" in m for m in config.model.edit_modules) and any("decoder" in m for m in config.model.edit_modules):
            
            for module_name in config.model.edit_modules:
                if "encoder" in module_name:
                    cache_mask = tuples["attention_mask"]
                else:
                    cache_mask = tuples["decoder_attention_mask"]
                module = get_module(model, module_name)
                self[module_name] = Tracer(module, cache_mask)

        else:

            if config.editor.token == "ans":
                cache_mask = tuples["labels"] != -100
            else:
                cache_mask = tuples["attention_mask"]

            for module_name in config.model.edit_modules:
                module = get_module(model, module_name)
                self[module_name] = Tracer(module, cache_mask)
            
    def __enter__(self):
        return self
            
    def __exit__(self, type, value, traceback):
        for v in self.values():
            for h in v.handles:
                h.remove()


class EarlyStopper:
    def __init__(self, patience: int, key: str):
        self.best_value = 1e9
        self.best_iter = 0
        self.current_iter = 0
        self.key = key
        self.patience = patience
        self._stop = False

    def update(self, idx, stats):
        assert self.key in stats, f"'{self.key}' not in stats dict"
        value = stats[self.key]
        new_best = value < self.best_value
        if new_best:
            self.best_value = value
            self.best_iter = idx

        self.current_iter = idx
        return new_best

    def should_stop(self):
        self._stop |= self.current_iter - self.best_iter >= self.patience
        return self._stop
