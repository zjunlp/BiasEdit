from typing import Dict

from data.base import BaseDataset

import  json, string, random, logging, copy, random, torch
from torch.utils.data import Dataset
from tqdm import tqdm
import copy
import math

class StereoSetDataset(Dataset):
    
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


        data = json.load(open(data_path))
        for d in data:
            self.data.append({k: d[k] for k in ["id", "target", "bias_type", "context", "data"]})
        # random.shuffle(self.data)
        
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        b = self.data[item]
        return {
            "context": b["context"],
            "anti-stereotype": b["data"]["anti-stereotype"]['sentence'],
            "stereotype": b["data"]["stereotype"]['sentence'],
            "unrelated": b["data"]["unrelated"]['sentence'],
        }
    
    def collate_fn(self, batch):
        new_batch = []
        for b in batch:
            word_idx = None
            blank_cnt = 0
            strange = False
            for idx, word in enumerate(b["context"].split(" ")):
                if "BLANK" in word: 
                    word_idx = idx
                    blank_cnt += 1
                if ".BLANK" in word or "`BLANK" in word:
                    strange = True
            if strange:
                print(b['context'])
                raise Exception("Noise.")
            if blank_cnt > 1:
                print(b['context'])
                raise Exception("Noise.")
            if word_idx is None:
                raise Exception("No BLANK found.")
            
            anti_word = b['anti-stereotype'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
            stereo_word = b['stereotype'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
            unrelated_word = b['unrelated'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
            
            if not self.iscausal:   # roberta
                # if "roberta" in self._tokenizer.__class__.__name__.lower():
                if b['context'].startswith("BLANK"):
                    anti_insertion_tokens = self._tokenizer.encode(anti_word, add_special_tokens=False)
                    stereo_insertion_tokens = self._tokenizer.encode(stereo_word, add_special_tokens=False)
                    unrelated_insertion_tokens = self._tokenizer.encode(unrelated_word, add_special_tokens=False)
                    anti_sentence = b['context'].replace("BLANK", self.mask_token * len(anti_insertion_tokens))
                    stereo_sentence = b['context'].replace("BLANK", self.mask_token * len(stereo_insertion_tokens))
                    unrelated_sentence = b['context'].replace("BLANK", self.mask_token * len(unrelated_insertion_tokens))
                else:
                    anti_word = " " + anti_word
                    stereo_word = " " + stereo_word
                    unrelated_word = " " + unrelated_word
                    anti_insertion_tokens = self._tokenizer.encode(anti_word, add_special_tokens=False)
                    stereo_insertion_tokens = self._tokenizer.encode(stereo_word, add_special_tokens=False)
                    unrelated_insertion_tokens = self._tokenizer.encode(unrelated_word, add_special_tokens=False)
                    anti_sentence = b['context'].replace(" BLANK", self.mask_token * len(anti_insertion_tokens))
                    stereo_sentence = b['context'].replace(" BLANK", self.mask_token * len(stereo_insertion_tokens))
                    unrelated_sentence = b['context'].replace(" BLANK", self.mask_token * len(unrelated_insertion_tokens))
                # else:   # bert
                #     anti_insertion_tokens = self._tokenizer.encode(anti_word, add_special_tokens=False)
                #     stereo_insertion_tokens = self._tokenizer.encode(stereo_word, add_special_tokens=False)
                #     unrelated_insertion_tokens = self._tokenizer.encode(unrelated_word, add_special_tokens=False)
                #     anti_sentence = b['context'].replace("BLANK", self.mask_token * len(anti_insertion_tokens))
                #     stereo_sentence = b['context'].replace("BLANK", self.mask_token * len(stereo_insertion_tokens))
                #     unrelated_sentence = b['context'].replace("BLANK", self.mask_token * len(unrelated_insertion_tokens))
                b['template_word_ids'] = {"anti": anti_insertion_tokens, "stereo": stereo_insertion_tokens, "unrelated": unrelated_insertion_tokens}
            else:
                if "gpt" in self._tokenizer.__class__.__name__.lower():
                    anti_sentence = self._tokenizer.bos_token + b['context'].replace("BLANK", anti_word)
                    stereo_sentence = self._tokenizer.bos_token + b['context'].replace("BLANK", stereo_word)
                    unrelated_sentence = self._tokenizer.bos_token + b['context'].replace("BLANK", unrelated_word)
                else: # llama, mistral
                    anti_sentence = b['context'].replace("BLANK", anti_word)
                    stereo_sentence = b['context'].replace("BLANK", stereo_word)
                    unrelated_sentence = b['context'].replace("BLANK", unrelated_word)
            b["input_sentence"] = {"anti": anti_sentence, "stereo": stereo_sentence, "unrelated": unrelated_sentence}   
            # b["insertion_tokens"] = {"anti": anti_insertion_tokens, "stereo": stereo_insertion_tokens, "unrelated": unrelated_insertion_tokens}
            new_batch.append(b)

        edit = []
        edit_unrelated = []

        if len(new_batch) < self.config.n_edits:
            for n_batch in range(math.ceil(len(new_batch) / self.config.batch_size)):
                if (n_batch + 1) * self.config.batch_size < len(new_batch):
                    batches = self.tok_samples(new_batch[n_batch * self.config.batch_size : (n_batch + 1) * self.config.batch_size])
                else:
                    batches = self.tok_samples(new_batch[n_batch * self.config.batch_size : ])
                edit.append(batches['edit'].to(self.device))
                edit_unrelated.append(batches['unrelated'].to(self.device))
        else:
            for n_batch in range(math.ceil(self.config.n_edits / self.config.batch_size)):
                if (n_batch + 1) * self.config.batch_size < len(new_batch):
                    batches = self.tok_samples(new_batch[n_batch * self.config.batch_size : (n_batch + 1) * self.config.batch_size])
                else:
                    batches = self.tok_samples(new_batch[n_batch * self.config.batch_size : ])
                edit.append(batches['edit'].to(self.device))
                edit_unrelated.append(batches['unrelated'].to(self.device))
        
        return {"edit": edit, "unrelated": edit_unrelated}

        
    def tok_samples(self, new_batch):
        batches = {}
        if not self.iscausal:   # MLM: roberta, bert
            # anti + stereo
            anti_srcs = [b["input_sentence"]["anti"] for b in new_batch]
            stereo_srcs = [b["input_sentence"]["stereo"] for b in new_batch]
            edit = anti_srcs + stereo_srcs  # config.data.batch_size * 2
            inputs = self._tokenizer(
                edit,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            inputs['labels'] = copy.deepcopy(inputs['input_ids'])
            for batchin_idx, input_ids in enumerate(inputs['labels']):
                if batchin_idx < len(new_batch):
                    mask_idxs = []
                    for mask_idx, input_id  in enumerate(input_ids):
                        if input_id == self.mask_token_id:
                            mask_idxs.append(mask_idx)
                    assert len(mask_idxs) == len(new_batch[batchin_idx]['template_word_ids']['anti'])
                    for template_idx, mask_idx in enumerate(mask_idxs):
                        inputs['labels'][batchin_idx][mask_idx] = new_batch[batchin_idx]['template_word_ids']['anti'][template_idx]
                    inputs['labels'][batchin_idx] = torch.where(inputs['input_ids'][batchin_idx] == self._tokenizer.mask_token_id, inputs['labels'][batchin_idx], -100)
                else:
                    new_batchin_idx = batchin_idx - len(new_batch)
                    mask_idxs = []
                    for mask_idx, input_id  in enumerate(input_ids):
                        if input_id == self.mask_token_id:
                            mask_idxs.append(mask_idx)
                    assert len(mask_idxs) == len(new_batch[new_batchin_idx]['template_word_ids']['stereo'])
                    for template_idx, mask_idx in enumerate(mask_idxs):
                        inputs['labels'][batchin_idx][mask_idx] = new_batch[new_batchin_idx]['template_word_ids']['stereo'][template_idx]
                    inputs['labels'][batchin_idx] = torch.where(inputs['input_ids'][batchin_idx] == self._tokenizer.mask_token_id, inputs['labels'][batchin_idx], -100)

            batches['edit'] = inputs

            # unrelated
            unrelated_srcs = [b["input_sentence"]["unrelated"] for b in new_batch]
            unrelated_inputs = self._tokenizer(
                unrelated_srcs,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            unrelated_inputs['labels'] = copy.deepcopy(unrelated_inputs['input_ids'])
            for batchin_idx, input_ids in enumerate(unrelated_inputs['labels']):
                mask_idxs = []
                for mask_idx, input_id in enumerate(input_ids):
                    if input_id == self.mask_token_id:
                        mask_idxs.append(mask_idx)
                assert len(mask_idxs) == len(new_batch[batchin_idx]['template_word_ids']['unrelated'])
                for template_idx, mask_idx in enumerate(mask_idxs):
                    unrelated_inputs['labels'][batchin_idx][mask_idx] = new_batch[batchin_idx]['template_word_ids']['unrelated'][template_idx]
                unrelated_inputs['labels'][batchin_idx] = torch.where(unrelated_inputs['input_ids'][batchin_idx] == self._tokenizer.mask_token_id, unrelated_inputs['labels'][batchin_idx], -100)
            batches['unrelated'] = unrelated_inputs
        else:   # CLM: gpt2, Llama
            anti_srcs = [b["input_sentence"]["anti"] for b in new_batch]
            stereo_srcs = [b["input_sentence"]["stereo"] for b in new_batch]
            edit = anti_srcs + stereo_srcs  # config.data.batch_size * 2
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

            # unrelated
            unrelated_srcs = [b["input_sentence"]["unrelated"] for b in new_batch]
            unrelated_inputs = self._tokenizer(
                unrelated_srcs,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            unrelated_inputs['labels'] = copy.deepcopy(unrelated_inputs['input_ids'])
            for batchin_idx in range(len(unrelated_inputs['labels'])):
                unrelated_inputs['labels'][batchin_idx] = torch.where(unrelated_inputs['labels'][batchin_idx]!=self._tokenizer.pad_token_id, unrelated_inputs['labels'][batchin_idx], -100)
                unrelated_inputs['labels'][batchin_idx][0] = -100
            batches['unrelated'] = unrelated_inputs
        return batches
