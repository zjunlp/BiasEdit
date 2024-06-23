from typing import Dict, List
from omegaconf import DictConfig

from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nets import MALMENNet

from tqdm import tqdm
import wandb, os, logging

from util import (
    get_module,
    get_shape,
    empty_cache,
    TracerDict,
    cross_entropy,
    kl_div,
    succ_ratios,
    EarlyStopper
)

LOG = logging.getLogger(__name__)

class BaseEditor:

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        
        self.config = config
        self.model = model

        self.modelname = model.__class__.__name__.lower()
        self.iscausal = "gpt" in self.modelname or "llama" in self.modelname or "mistral" in self.modelname or "gemma" in self.modelname
        if self.iscausal:
            self.loc_loss_fn = self._loc_causal_loss_fn
            self.edit_loss_fn = self._edit_causal_loss_fn
        else:
            self.loc_loss_fn = self._loc_loss_fn
            self.edit_loss_fn = self._edit_loss_fn
        
        shape_counter = Counter()
        self.name2idx = {}
        for module_name in config.model.edit_modules:
            shape = get_shape(get_module(model, module_name))
            self.name2idx[module_name] = shape_counter[shape]
            shape_counter[shape] += 1

        self.net = nn.ModuleDict({
            str(k): MALMENNet(
                *k,
                config.editor.rank,
                config.editor.n_blocks,
                v,
                config.editor.lr
            )
            for k, v in shape_counter.items()
        }).to(config.editor_device)
        
        self.opt = torch.optim.Adam(
            self.net.parameters(),
            config.editor.meta_lr
        )
        if config.editor.load_checkpoint:
            self.net.load_state_dict(torch.load(f"checkpoints/{str(self.config.model.layers)}_{str(self.config.data.n_edits)}_net.pth"))
            self.opt.load_state_dict(torch.load(f"checkpoints/{str(self.config.model.layers)}_{str(self.config.data.n_edits)}_opt.pth"))

    def edit_model(
        self,
        param_shifts: Dict[str, torch.FloatTensor],
        is_reverse: bool
    ):
        
        for module_name, param_shift in param_shifts.items():
            module = get_module(self.model, module_name)
            if isinstance(module, nn.Linear):
                param_shift = param_shift.T
            if is_reverse:
                param_shift = - param_shift
            module.weight.data += param_shift.to(module.weight.data.dtype)

    def train(self, loader: DataLoader):

        cnt = 0
        info_dicts = []
        for tuples in tqdm(loader, desc = "Train", ncols = 100):    # len(loader) = len(data) / data.n_edits
            cnt += 1
            pre_edit_dicts = self.cache(tuples)   # cache (u, v_grad)
            param_shifts = self.predict_param_shifts()  # Feed (u, v_grad) into hyper-network to infer \tilde{W}
            self.model.zero_grad()

            gen_losses = []
            post_edit_dicts = []
            self.edit_model(param_shifts, False)    # edit
            for t in tuples["edit"]:
                post_edit_logits = self.model(**t).logits
                assert post_edit_logits.shape[0] % 2 == 0
                post_edit_logits_anti = post_edit_logits[:(post_edit_logits.shape[0] // 2)]
                post_edit_logits_stereo = post_edit_logits[(post_edit_logits.shape[0] // 2):]
                post_edit_dict = self.edit_loss_fn(post_edit_logits_anti, t["labels"][:(post_edit_logits.shape[0] // 2)], post_edit_logits_stereo, t["labels"][(post_edit_logits.shape[0] // 2):])
                post_edit_dicts.append(post_edit_dict)
                post_edit_loss = post_edit_dict['loss']
                post_edit_loss.backward()
                gen_losses += [post_edit_loss.item()]
            
            self.edit_model(param_shifts, True) # recover

            loc_losses = []
            pre_loc_dicts = []
            post_loc_dicts = []
            for t in tuples["unrelated"]:
                with torch.no_grad():
                    base_output = self.model(**t, return_dict=True)
                pre_loc_dict = self.loc_loss_fn(base_output, t["labels"])
                pre_loc_dicts.append(pre_loc_dict)

                self.edit_model(param_shifts, False)    # edit
                post_loc_output = self.model(**t)
                post_loc_dict = self.loc_loss_fn(post_loc_output, t["labels"])
                post_loc_dicts.append(post_loc_dict)
                kl_loss = nn.KLDivLoss(reduction="batchmean")
                loc_loss = kl_loss(input=post_loc_dict['loc_log_score'], target=pre_loc_dict['loc_score'])

                (self.config.editor.loc_coef * loc_loss).backward()
                self.edit_model(param_shifts, True)     # recover
                loc_losses += [loc_loss.item()]
                
            self.update_hypernet(param_shifts)

            #prelms
            prelmses = []
            for idx in range(len(pre_edit_dicts)):
                prelmses.append(self.lms(pre_edit_dicts[idx], pre_loc_dicts[idx]))

            editlmses = []
            for idx in range(len(post_edit_dicts)):
                editlmses.append(self.lms(post_edit_dicts[idx], post_loc_dicts[idx]))
            
            info_dict = {}
            info_dict["train_pre/ss_score"] = np.mean([ss['ss_score'] for ss in pre_edit_dicts])
            info_dict["train_edit/ss_score"] = np.mean([ss['ss_score'] for ss in post_edit_dicts])
            info_dict["train_pre/lms"] = np.mean(prelmses)
            info_dict["train_edit/lms"] = np.mean(editlmses)
            info_dict['train_edit/loc'] = np.mean(loc_losses)
            info_dict['train_edit/gen_loss'] = np.mean(gen_losses)
            info_dict['train_edit/loss'] = info_dict['train_edit/gen_loss'] + self.config.editor.loc_coef * info_dict['train_edit/loc']
            info_dicts.append(info_dict)
            print(f"Train {cnt} -------- pre_ss: {info_dict['train_pre/ss_score']}, edit_ss: {info_dict['train_edit/ss_score']}, pre_lms: {info_dict['train_pre/lms']}, edit_lms: {info_dict['train_edit/lms']}, delta_lms: {info_dict['train_edit/lms']-info_dict['train_pre/lms']}")
        if self.config.use_wandb:
            wandb.log(info_dict)
        
        return info_dicts
    
    def valid(self, loader: DataLoader):
        info_dicts = []
        cnt = 0
        if self.config.eval_only and self.config.save_testckpt:
            ckptdir = os.path.join("checkpoints", self.config.model.layers, self.config.data.valid_path.split('/')[-1].split('_')[0])
            f"checkpoints/{self.config.model.layers}"
            os.makedirs(ckptdir, exist_ok=True)
        for tuples in tqdm(loader, desc = "Valid", ncols = 100):
            cnt += 1
            pre_edit_dicts = self.cache(tuples)
            # print(torch.cuda.memory_allocated())
            param_shifts = self.predict_param_shifts()
            self.edit_model(param_shifts, False)    # edit
            if self.config.eval_only and self.config.save_testckpt:
                if "race" in self.config.data.valid_path:
                    if cnt % 4 ==0:
                        self.model.save_pretrained(os.path.join(ckptdir, f"Test_{cnt}"))
                else:
                    self.model.save_pretrained(os.path.join(ckptdir, f"Test_{cnt}"))
            gen_losses = []
            post_edit_dicts = []
            for t in tuples["edit"]:
                with torch.no_grad():
                    post_edit_logits = self.model(**t).logits
                assert post_edit_logits.shape[0] % 2 == 0
                post_edit_logits_anti = post_edit_logits[:(post_edit_logits.shape[0] // 2)]
                post_edit_logits_stereo = post_edit_logits[(post_edit_logits.shape[0] // 2):]
                post_edit_dict = self.edit_loss_fn(post_edit_logits_anti, t["labels"][:(post_edit_logits.shape[0] // 2)], post_edit_logits_stereo, t["labels"][(post_edit_logits.shape[0] // 2):])
                post_edit_dicts.append(post_edit_dict)
                gen_losses += [post_edit_dict['loss'].item()]

            self.edit_model(param_shifts, True) # recover
            
            loc_losses = []
            pre_loc_dicts = []
            post_loc_dicts = []
            for t in tuples["unrelated"]:
                with torch.no_grad():
                    base_output = self.model(**t, return_dict=True)
                pre_loc_dict = self.loc_loss_fn(base_output, t["labels"])
                pre_loc_dicts.append(pre_loc_dict)

                self.edit_model(param_shifts, False)    # edit
                with torch.no_grad():
                    post_loc_output = self.model(**t)
                post_loc_dict = self.loc_loss_fn(post_loc_output, t["labels"])
                post_loc_dicts.append(post_loc_dict)
                kl_loss = nn.KLDivLoss(reduction="batchmean")
                loc_loss = kl_loss(input=post_loc_dict['loc_log_score'], target=pre_loc_dict['loc_score'])

                self.edit_model(param_shifts, True)     # recover
                loc_losses += [loc_loss.item()]

            #prelms
            prelmses = []
            for idx in range(len(pre_edit_dicts)):
                prelmses.append(self.lms(pre_edit_dicts[idx], pre_loc_dicts[idx]))
            #postlms
            editlmses = []
            for idx in range(len(post_edit_dicts)):
                editlmses.append(self.lms(post_edit_dicts[idx], post_loc_dicts[idx]))
            
            info_dict = {}
            info_dict["val_pre/ss_score"] = np.mean([ss['ss_score'] for ss in pre_edit_dicts])
            info_dict["val_edit/ss_score"] = np.mean([ss['ss_score'] for ss in post_edit_dicts])
            info_dict["val_pre/lms"] = np.mean(prelmses)
            info_dict["val_edit/lms"] = np.mean(editlmses)
            info_dict['val_edit/loc'] = np.mean(loc_losses)
            info_dict['val_edit/gen_loss'] = np.mean(gen_losses)
            info_dict['val_edit/loss'] = info_dict['val_edit/gen_loss'] + self.config.editor.loc_coef * info_dict['val_edit/loc']
            info_dicts.append(info_dict)
            if self.config.eval_only:
                LOG.info(f"Test {cnt} -------- pre_ss: {info_dict['val_pre/ss_score']}, edit_ss: {info_dict['val_edit/ss_score']}, pre_lms: {info_dict['val_pre/lms']}, edit_lms: {info_dict['val_edit/lms']}, delta_lms: {info_dict['val_edit/lms']-info_dict['val_pre/lms']}")
            else:
                LOG.info(f"Valid {cnt} -------- pre_ss: {info_dict['val_pre/ss_score']}, edit_ss: {info_dict['val_edit/ss_score']}, pre_lms: {info_dict['val_pre/lms']}, edit_lms: {info_dict['val_edit/lms']}, delta_lms: {info_dict['val_edit/lms']-info_dict['val_pre/lms']}")
            
            if self.config.use_wandb:
                wandb.log(info_dict)
        all_pre_ss = sum([info_dict['val_pre/ss_score'] for info_dict in info_dicts]) / len(info_dicts)
        all_edit_ss = sum([info_dict['val_edit/ss_score'] for info_dict in info_dicts]) / len(info_dicts)
        all_pre_lms = sum([info_dict['val_pre/lms'] for info_dict in info_dicts]) / len(info_dicts)
        all_edit_lms = sum([info_dict['val_edit/lms'] for info_dict in info_dicts]) / len(info_dicts)
        all_edit_loss = sum([info_dict['val_edit/loss'] for info_dict in info_dicts]) / len(info_dicts)
        if self.config.eval_only:
            LOG.info(f"Overall results: \n Test -------- pre_ss: {all_pre_ss}, edit_ss: {all_edit_ss}, pre_lms: {all_pre_lms}, edit_lms: {all_edit_lms}, delta_lms: {all_edit_lms - all_pre_lms}")
        else:
            LOG.info(f"Overall results: \n Valid -------- pre_ss: {all_pre_ss}, edit_ss: {all_edit_ss}, pre_lms: {all_pre_lms}, edit_lms: {all_edit_lms}, delta_lms: {all_edit_lms - all_pre_lms}")
        
        
        return {
            "pre/ss_score": all_pre_ss,
            "edit/ss_score": all_edit_ss,
            "pre/lms": all_pre_lms,
            "edit/lms": all_edit_lms,
            "edit/loss": all_edit_loss
        }
    
    def cache(self, tuples): # cache (u, v_grad)
        cache_dir = os.path.join(self.config.editor.cache_dir, self.config.model.layers)
        os.makedirs(cache_dir, exist_ok=True)
        edit_dicts = []
        for idx in range(len(tuples['edit'])):
            # data.batch_size
            with TracerDict(
                self.model,
                self.config,
                tuples['edit'][idx],
            ) as tr:
                logits = self.model(**tuples['edit'][idx], return_dict=True).logits
                assert logits.shape[0] % 2 == 0
                logits_anti = logits[:(logits.shape[0] // 2)]
                logits_stereo = logits[(logits.shape[0] // 2):]
                edit_dict = self.edit_loss_fn(logits_anti, tuples['edit'][idx]["labels"][:(logits.shape[0] // 2)], logits_stereo, tuples['edit'][idx]["labels"][(logits.shape[0] // 2):])
                edit_dict['loss'].backward()
                edit_dicts.append(edit_dict)
        
            for module_idx, module_name in enumerate(self.config.model.edit_modules):
                shape = get_shape(get_module(self.model, module_name))
                keys = tr[module_name].keys.to(torch.float32).to(self.config.editor_device)
                values_grad = tr[module_name].values_grad.to(torch.float32).to(self.config.editor_device)
                self.net[str(shape)].normalizer.update(torch.cat((keys, values_grad), -1))
                torch.save(keys, os.path.join(cache_dir, f"{module_idx}_{idx}_keys.pth"))
                torch.save(values_grad, os.path.join(cache_dir, f"{module_idx}_{idx}_values_grad.pth"))

        return edit_dicts
                
    
    def run(self, train_loader: DataLoader, valid_loader: DataLoader):
        
        empty_cache(f"{self.config.editor.cache_dir}/{self.config.model.layers}")
        os.makedirs("checkpoints", exist_ok=True)
        stopper = EarlyStopper(self.config.early_stop_patience, self.config.early_stop_key)
        for _ in range(self.config.editor.n_epochs):
            train_result = self.train(train_loader)
            val_result = self.valid(valid_loader) 
            if stopper.update(_, val_result):
                torch.save(self.net.state_dict(), f"checkpoints/{str(self.config.model.layers)}_{str(self.config.data.n_edits)}_net.pth")
                torch.save(self.opt.state_dict(), f"checkpoints/{str(self.config.model.layers)}_{str(self.config.data.n_edits)}_opt.pth")  # New best
            if stopper.should_stop():
                LOG.info(f"No decrease in {self.config.early_stop_key} for {self.config.early_stop_patience} epochs")
                break
            
                   
    
    def _loc_loss_fn(self, output, targ):
        output_soft = F.softmax(output.logits, dim=-1)
        output_log_soft = F.log_softmax(output.logits, dim=-1)
        n_tokens = 0

        def get_loc(p,t):
            cnt_tokens = 0
            mask_idxs = (t != -100)
            template_word_idxs = []
            for i in range(len(t)):
                if t[i] != -100:
                    template_word_idxs.append(t[i])
                    cnt_tokens += 1
            ele_probs = p[mask_idxs]
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(output_soft.device)).diag()
            return ele_probs.mean(), cnt_tokens
        
        loc_score, cnt = get_loc(output_soft[0], targ[0])
        loc_score = loc_score.unsqueeze(0)
        n_tokens += cnt
        for batch_idx, (p, t) in enumerate(zip(output_soft, targ)):
            if batch_idx==0:
                continue
            probs, cnt_token = get_loc(p, t)
            n_tokens += cnt_token
            loc_score = torch.cat((loc_score, probs.unsqueeze(0)), dim=0)
        
        loc_log_score, _ = get_loc(output_log_soft[0], targ[0])
        loc_log_score = loc_log_score.unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(output_log_soft, targ)):
            if batch_idx==0:
                continue
            probs, _ = get_loc(p, t)
            loc_log_score = torch.cat((loc_log_score, probs.unsqueeze(0)), dim=0)

        return {
            # "avg_prob": loc_score.mean(),
            "loss": output.loss,
            "n_tokens": n_tokens, 
            "loc_score": loc_score,
            "loc_log_score": loc_log_score
        }

    
    def _edit_loss_fn(self, pred_anti, targ_anti, pred_stereo, targ_stereo):   # pred_anti: (batch_size, seq_len, vocab_size), targ_anti: (batch_size, seq_len)        
        
        def get_edit(p,t):
            mask_idxs = (t != -100)
            template_word_idxs = [] # ids of [MASK] target tokens
            for i in range(len(t)): # seq_len
                if t[i] != -100:
                    template_word_idxs.append(t[i])
                    # anti_n_tokens += 1
            ele_probs = p.softmax(-1)[mask_idxs] # [MASK] probs, (mask_num, vocab_size)
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(pred_anti.device)).diag()   # [MASK] target tokens probs (mask_num,)
            return ele_probs.mean()
        
        def get_log_edit(p,t):
            mask_idxs = (t != -100)
            template_word_idxs = [] # ids of [MASK] target tokens
            for i in range(len(t)): # seq_len
                if t[i] != -100:
                    template_word_idxs.append(t[i])
                    # anti_n_tokens += 1
            ele_probs = p.log_softmax(-1)[mask_idxs] # [MASK] probs, (mask_num, vocab_size)
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(pred_anti.device)).diag()   # [MASK] target tokens probs (mask_num,)
            return ele_probs.mean()
        
        anti_score = get_edit(pred_anti[0], targ_anti[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_anti, targ_anti)):
            if batch_idx==0:
                continue
            anti_score = torch.cat((anti_score, get_edit(p,t).unsqueeze(0)), dim=0)                                              # mean probs of [MASK] target tokens
        
        stereo_score = get_edit(pred_stereo[0], targ_stereo[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_stereo, targ_stereo)):
            if batch_idx==0:
                continue
            stereo_score = torch.cat((stereo_score, get_edit(p,t).unsqueeze(0)), dim=0)                                              # mean probs of [MASK] target tokens
        
        anti_log_score = get_log_edit(pred_anti[0], targ_anti[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_anti, targ_anti)):
            if batch_idx==0:
                continue
            anti_log_score = torch.cat((anti_log_score, get_log_edit(p,t).unsqueeze(0)), dim=0)                                              # mean probs of [MASK] target tokens
        
        stereo_log_score = get_log_edit(pred_stereo[0], targ_stereo[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_stereo, targ_stereo)):
            if batch_idx==0:
                continue
            stereo_log_score = torch.cat((stereo_log_score, get_log_edit(p, t).unsqueeze(0)), dim=0)              
        
        # kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        # loss = kl_loss(input=anti_log_score, target=stereo_log_score) + kl_loss(input=stereo_log_score, target=anti_log_score)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = kl_loss(input=anti_log_score, target=stereo_score) + kl_loss(input=stereo_log_score, target=anti_score)
        
        # SS
        stereo_preferred, anti_preferred, neither_preferred = 0, 0, 0
        for anti, pro in zip(anti_log_score, stereo_log_score):
            if pro > anti:
                stereo_preferred += 1
            elif anti > pro:
                anti_preferred += 1
            else:
                neither_preferred += 1
        
        return {
            "ss_score": 50.00 if stereo_preferred + anti_preferred==0.00 else stereo_preferred / (stereo_preferred + anti_preferred),
            "loss": loss,
            # "anti_prob": anti_score.mean(),
            # "stereo_prob": stereo_score.mean(),
            "anti_score": anti_score, 
            "stereo_score": stereo_score,
            "anti_log_score": anti_log_score,
            "stereo_log_score": stereo_log_score
        }
    
    def _edit_causal_loss_fn(self, pred_anti, targ_anti, pred_stereo, targ_stereo, shift=True):

        if shift and pred_anti.dim() == 3 and pred_stereo.dim() == 3:  # Dealing with sequences
            pred_anti = pred_anti[:, :-1]  # Remove last prediction in sequence
            targ_anti = targ_anti[:, 1:]  # Shift to align predictions and targets
            pred_stereo = pred_stereo[:, :-1]
            targ_stereo = targ_stereo[:, 1:]

        def get_score(p,t, log=True):
            mask_idxs = (t != -100)
            template_word_idxs = []
            for i in range(len(t)):
                if t[i] != -100:
                    template_word_idxs.append(t[i])
            if log:
                ele_probs = p.log_softmax(-1)[mask_idxs]
            else:
                ele_probs = p.softmax(-1)[mask_idxs]
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(pred_anti.device)).diag()
            return ele_probs.mean()
        
        anti_score = get_score(pred_anti[0], targ_anti[0], False).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_anti, targ_anti)):
            if batch_idx==0:
                continue
            anti_score = torch.cat((anti_score, get_score(p, t, False).unsqueeze(0)), dim=0)   # (batch_size, )
          

        stereo_score = get_score(pred_stereo[0], targ_stereo[0], False).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_stereo, targ_stereo)):
            if batch_idx==0:
                continue
            stereo_score = torch.cat((stereo_score, get_score(p, t, False).unsqueeze(0)), dim=0)   # (batch_size, )
        
        anti_log_score = get_score(pred_anti[0], targ_anti[0], True).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_anti, targ_anti)):
            if batch_idx==0:
                continue
            anti_log_score = torch.cat((anti_log_score, get_score(p, t, True).unsqueeze(0)), dim=0)
        
        stereo_log_score = get_score(pred_stereo[0], targ_stereo[0], True).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_stereo, targ_stereo)):
            if batch_idx==0:
                continue
            stereo_log_score = torch.cat((stereo_log_score, get_score(p, t, True).unsqueeze(0)), dim=0)

        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = kl_loss(input=anti_log_score, target=stereo_score) + kl_loss(input=stereo_log_score, target=anti_score)
        pro_num = 0
        total = 0
        for anti, pro in zip(anti_score, stereo_score):
            if pro > anti:
                pro_num += 1
            total += 1
        ss_score = pro_num / total
        return {
            "ss_score": ss_score,
            "loss": loss,
            # "anti_prob": anti_score.mean(),
            # "stereo_prob": stereo_score.mean(),
            "anti_score": anti_score,
            "stereo_score": stereo_score,
            "anti_log_score": anti_log_score,
            "stereo_log_score": stereo_log_score
        }
    
    

    def _loc_causal_loss_fn(self, output, targ, shift=True):
        pred = output.logits
        target = targ
        if shift and pred.dim() == 3:
            pred = pred[:, :-1]  # Remove last prediction in sequence
            target = target[:, 1:]  

        total_tokens = 0
        def get_score_loc(p,t, log=True):
            cnt_tokens = 0
            mask_idxs = (t != -100)
            template_word_idxs = []
            for i in range(len(t)):
                if t[i] != -100:
                    template_word_idxs.append(t[i])
                    cnt_tokens += 1
            if log:
                ele_probs = p.log_softmax(-1)[mask_idxs]
            else:
                ele_probs = p.softmax(-1)[mask_idxs]
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(output.logits.device)).diag()
            return ele_probs.mean(), cnt_tokens

        loc_score, cnt = get_score_loc(pred[0], target[0], False)
        loc_score = loc_score.unsqueeze(0)
        total_tokens += cnt
        for batch_idx, (p, t) in enumerate(zip(pred, target)):
            if batch_idx==0:
                continue
            probs, cnt_token = get_score_loc(p, t, False)
            total_tokens += cnt_token
            loc_score = torch.cat((loc_score, probs.unsqueeze(0)), dim=0)   # (batch_size, )
        
        loc_log_score, _ = get_score_loc(pred[0], target[0], True)
        loc_log_score = loc_log_score.unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred, target)):
            if batch_idx==0:
                continue
            probs, _ = get_score_loc(p, t, True)
            loc_log_score = torch.cat((loc_log_score, probs.unsqueeze(0)), dim=0) # (batch_size, )

        return {
            # "avg_prob": loc_score.mean(),
            "loss": output.loss,
            "n_tokens": total_tokens,
            "loc_score": loc_score,
            "loc_log_score": loc_log_score
        }

    def _logits(x):
        return x if not hasattr(x, "logits") else x.logits
    
    def lms(self, edit_dict, loc_dict):
        relevant_preferred, total_preferred = 0, 0
        if self.iscausal:
            for antis, locs in zip(edit_dict['anti_log_score'], loc_dict['loc_log_score']):
                if antis > locs:
                    relevant_preferred += 1
                total_preferred += 1
            for stereos, locs in zip(edit_dict['stereo_log_score'], loc_dict['loc_log_score']):
                if stereos > locs:
                    relevant_preferred += 1
                total_preferred += 1
        else:
            for antis, locs in zip(edit_dict['anti_score'], loc_dict['loc_score']):
                if antis > locs:
                    relevant_preferred += 1
                total_preferred += 1
            for stereos, locs in zip(edit_dict['stereo_score'], loc_dict['loc_score']):
                if stereos > locs:
                    relevant_preferred += 1
                total_preferred += 1
        lms = relevant_preferred / total_preferred

        return lms