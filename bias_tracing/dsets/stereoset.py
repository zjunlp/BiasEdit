
# from pathlib import Path
# import jsonlines, json, string, random, logging, copy
import numpy as np
import torch, json, string
from torch.utils.data import Dataset
# from utils import EditBatchSampler, dict_to, scr
from tqdm import tqdm
import copy

import torch
from torch.utils.data import Dataset

from util.globals import *

from transformers import (
    BertTokenizerFast,
    GPT2TokenizerFast,
    LlamaTokenizer
)

class StereoSetDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        # config,
        model_name,
        # max_length=64
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        # self.config = config
        self.ifcausal = "gpt" in model_name or "llama" in model_name
        if self.ifcausal:
            self.mask_token = tokenizer.unk_token
            self.mask_token_id = tokenizer.unk_token_id
        else:
            self.mask_token = tokenizer.mask_token
            self.mask_token_id = tokenizer.mask_token_id

        data = json.load(open(data_path))
        for d in data:
            self.data.append({k: d[k] for k in ["id", "target", "bias_type", "context", "data", "subject"]})
        
        # self.max_length = max_length

        print(f"Loaded dataset with {len(self)} elements")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        obj = self.data[item]
        word_idx = None
        for idx, word in enumerate(obj["context"].split(" ")):
            if "BLANK" in word: 
                word_idx = idx
                break
        if word_idx is None:
            raise Exception("No blank word found.")
        
        anti_word = obj['data']['anti-stereotype']['sentence'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
        stereo_word = obj['data']['stereotype']['sentence'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
        unrelated_word = obj['data']['unrelated']['sentence'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))

        if "roberta" in self.tokenizer.__class__.__name__.lower():
            if word_idx!=0:
                anti_word = " " + anti_word
                stereo_word = " " + stereo_word
                unrelated_word = " " + unrelated_word
                anti_blank_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
                stereo_blank_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
                unrelated_blank_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
            else:
                anti_blank_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
                stereo_blank_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
                unrelated_blank_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
            anti_sentence = obj['context'].replace("BLANK", self.mask_token * len(anti_blank_tokens))
            stereo_sentence = obj['context'].replace("BLANK", self.mask_token * len(stereo_blank_tokens))
            unrelated_sentence = obj['context'].replace("BLANK", self.mask_token * len(unrelated_blank_tokens))
        elif "gpt" in self.tokenizer.__class__.__name__.lower():
            if " BLANK" in obj['context']:
                anti_word = " " + anti_word
                stereo_word = " " + stereo_word
                unrelated_word = " " + unrelated_word
                anti_blank_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
                stereo_blank_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
                unrelated_blank_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
                anti_sentence = obj['context'].replace(" BLANK", self.mask_token * len(anti_blank_tokens))
                stereo_sentence = obj['context'].replace(" BLANK", self.mask_token * len(stereo_blank_tokens))
                unrelated_sentence = obj['context'].replace(" BLANK", self.mask_token * len(unrelated_blank_tokens))
            else:
                anti_blank_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
                stereo_blank_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
                unrelated_blank_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
                anti_sentence = obj['context'].replace("BLANK", self.mask_token * len(anti_blank_tokens))
                stereo_sentence = obj['context'].replace("BLANK", self.mask_token * len(stereo_blank_tokens))
                unrelated_sentence = obj['context'].replace("BLANK", self.mask_token * len(unrelated_blank_tokens))      
        elif "llama" in self.tokenizer.__class__.__name__.lower():
            anti_blank_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
            stereo_blank_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
            unrelated_blank_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
            if " BLANK " in obj['context']:
                anti_sentence = obj['context'].replace(" BLANK ", self.mask_token * len(anti_blank_tokens))
                stereo_sentence = obj['context'].replace(" BLANK ", self.mask_token * len(stereo_blank_tokens))
                unrelated_sentence = obj['context'].replace(" BLANK ", self.mask_token * len(unrelated_blank_tokens))
            elif " BLANK" in obj['context']:
                anti_sentence = obj['context'].replace(" BLANK", self.mask_token * len(anti_blank_tokens))
                stereo_sentence = obj['context'].replace(" BLANK", self.mask_token * len(stereo_blank_tokens))
                unrelated_sentence = obj['context'].replace(" BLANK", self.mask_token * len(unrelated_blank_tokens))
            elif obj['context'].startwith("BLANK "):
                anti_sentence = obj['context'].replace("BLANK ", self.mask_token * len(anti_blank_tokens))
                stereo_sentence = obj['context'].replace("BLANK ", self.mask_token * len(stereo_blank_tokens))
                unrelated_sentence = obj['context'].replace("BLANK ", self.mask_token * len(unrelated_blank_tokens))
            else: # start with BLANK+punctuation
                anti_sentence = obj['context'].replace("BLANK", self.mask_token * len(anti_blank_tokens))
                stereo_sentence = obj['context'].replace("BLANK", self.mask_token * len(stereo_blank_tokens))
                unrelated_sentence = obj['context'].replace("BLANK", self.mask_token * len(unrelated_blank_tokens))
        else:   # bert
            anti_blank_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
            stereo_blank_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
            unrelated_blank_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
            anti_sentence = obj['context'].replace("BLANK", self.mask_token * len(anti_blank_tokens))
            stereo_sentence = obj['context'].replace("BLANK", self.mask_token * len(stereo_blank_tokens))
            unrelated_sentence = obj['context'].replace("BLANK", self.mask_token * len(unrelated_blank_tokens))

        output = {
            "id": obj['id'],
            "context": obj["context"],
            "anti": obj["data"]["anti-stereotype"]['sentence'],
            "stereo": obj["data"]["stereotype"]['sentence'],
            "unrelated": obj["data"]["unrelated"]['sentence'],
            "anti_mask": anti_sentence,
            "stereo_mask": stereo_sentence,
            "unrelated_mask": unrelated_sentence,
            "attribute": {"anti": anti_word, "stereo": stereo_word, "unrelated": unrelated_word},
            "target": obj['target'],
            "subject": obj['subject']
        }

        return output

    # def collate_fn(self, batch):
    #     for b in batch:
    #         word_idx = None
    #         for idx, word in enumerate(b["context"].split(" ")):
    #             if "BLANK" in word: 
    #                 word_idx = idx
    #                 break
    #         if word_idx is None:
    #             raise Exception("No blank word found.")
            
    #         anti_word = b['anti-stereotype'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
    #         stereo_word = b['stereotype'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
    #         unrelated_word = b['unrelated'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
            
    #         anti_blank_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
    #         stereo_blank_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
    #         unrelated_blank_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
            
    #         anti_sentence = b['context'].replace("BLANK", self.mask_token * len(anti_blank_tokens))
    #         stereo_sentence = b['context'].replace("BLANK", self.mask_token * len(stereo_blank_tokens))
    #         unrelated_sentence = b['context'].replace("BLANK", self.mask_token * len(unrelated_blank_tokens))
            
    #         b['template_word'] = {"anti": anti_word, "stereo": stereo_word, "unrelated": unrelated_word}
    #         b["input_sentence"] = {"anti": anti_sentence, "stereo": stereo_sentence, "unrelated": unrelated_sentence}
    #         b["blank_tokens"] = {"anti": anti_blank_tokens, "stereo": stereo_blank_tokens, "unrelated": unrelated_blank_tokens}

    #     if not self.ifcausal:
    #         anti_src = [b["input_sentence"]["anti"] for b in batch]
    #         stereo_src = [b["input_sentence"]["stereo"] for b in batch]
    #         unrelated_src = [b["input_sentence"]["unrelated"] for b in batch]
    #         anti_labels = [b["anti-stereotype"] for b in batch]
    #         stereo_labels = [b["stereotype"] for b in batch]
    #         unrelated_labels = [b["unrelated"] for b in batch]
    #     else:
    #         anti_src = [self.tokenizer.bos_token + b["input_sentence"]["anti"] for b in batch]
    #         stereo_src = [self.tokenizer.bos_token + b["input_sentence"]["stereo"] for b in batch]
    #         unrelated_src = [self.tokenizer.bos_token + b["input_sentence"]["unrelated"] for b in batch]
    #         anti_labels = [self.tokenizer.bos_token + b["anti-stereotype"] for b in batch]
    #         stereo_labels = [self.tokenizer.bos_token + b["stereotype"] for b in batch]
    #         unrelated_labels = [self.tokenizer.bos_token + b["unrelated"] for b in batch]

    #     res = [("anti", anti_src, "anti_labels", anti_labels), 
    #            ("stereo", stereo_src, "stereo_labels", stereo_labels),
    #            ("unrelated", unrelated_src, "unrelated_labels", unrelated_labels)]

    #     batches = {}
    #     for strsrc, srcs, labelstr, label in res:
    #         if not self.ifcausal:
    #             encoded = self.tokenizer(
    #                 srcs,
    #                 return_tensors="pt",
    #                 padding="max_length",
    #                 max_length=self.max_length,
    #                 truncation=True,
    #             )
    #             assert self.max_length == encoded['input_ids'].shape[1]
    #             labels = self.tokenizer(
    #                 label,
    #                 return_tensors="pt",
    #                 padding="max_length",
    #                 max_length=self.max_length,
    #                 truncation=True,
    #             )
    #             assert self.max_length == labels['input_ids'].shape[1]

    #             batches[f"{strsrc}"] = copy.deepcopy(dict(encoded.items()))

    #             for idx, input_ids in enumerate(labels["input_ids"]):
    #                 labels['input_ids'][idx] = torch.where(encoded['input_ids'][idx] == self.mask_token_id, input_ids, -100)
    #             batches[f"{strsrc}"]["labels"] = labels["input_ids"]

    #         else:
    #             encoded = self.tokenizer(
    #                 srcs,
    #                 return_tensors="pt",
    #                 padding="max_length",
    #                 max_length=self.max_length,
    #                 truncation=True,
    #             )
    #             assert self.max_length == encoded['input_ids'].shape[1]
    #             labels = self.tokenizer(
    #                 label,
    #                 return_tensors="pt",
    #                 padding="max_length",
    #                 max_length=self.max_length,
    #                 truncation=True,
    #             )
    #             batches[f"{strsrc}"] = copy.deepcopy(dict(labels.items()))
    #             for idx, input_ids in enumerate(labels["input_ids"]):
    #                 labels['input_ids'][idx] = torch.where(encoded['input_ids'][idx] != self.tokenizer.pad_token_id, input_ids, -100)
    #                 labels['input_ids'][idx][0] = -100
    #             batches[f"{strsrc}"]["labels"] = labels["input_ids"]

                
                

    #     batches["raw"] = batch
    #     return batches

    # def edit_generator(self, batch_size, n=None):
    #     if n is None:
    #         n = len(self)
    #     sampler = EditBatchSampler(n, memorize_mode=self.config.single_batch, seed=self.config.seed)
    #     while True:
    #         edit_idxs = sampler.stereosample(batch_size)

    #         toks = self.collate_fn([self[idx] for idx in edit_idxs])

    #         edit_inner = {"anti":toks['anti'], "stereo":toks['stereo']}

    #         edit_outer = edit_inner

    #         loc = toks["unrelated"]

    #         cond = None

    #         batch = {
    #             "edit_inner": edit_inner,
    #             "edit_outer": edit_outer,
    #             "loc": loc,
    #             "cond": cond
    #         }
    #         yield dict_to(batch, self.config.device)

# if __name__ == "__main__":
#     tokenizer = LlamaTokenizer.from_pretrained("llama-2-7b")
#     dataset = StereoSetDataset(tokenizer, "data/stereoset/test.json", None)
#     batch = [dataset[idx] for idx in [5,6,7,8]]
#     sample = next(dataset.edit_generator(4))
#     print(sample)

