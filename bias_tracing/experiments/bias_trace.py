import argparse
import json, copy
import os,sys
# os.chdir(sys.path[0])
sys.path.append("./")
import re
from collections import defaultdict

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForMaskedLM,
    AutoTokenizer, 
    LlamaTokenizer, 
    LlamaForCausalLM,
    GPT2Tokenizer,
    GPT2LMHeadModel
)

from dsets import KnownsDataset, StereoSetDataset
from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util.runningstats import Covariance, tally
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default="gpt2-medium",
        choices=[
            "llama-2-7b",
            "gpt2-xl",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
            "roberta-large",
            "bert-large-cased",
            "gpt-j-6b"
        ],
    )
    aa("--bias_file", default="data/stereoset/domain/gender.json")
    aa("--subject_file", default="data/knowns.json")          # set(stereoset target + stereoset subject + jieyu)
    aa("--output_dir", default="results/{model_name}/causal_trace")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    aa("--pattern", default="all", choices=["all", "one"], type=str)
    aa("--samples", default=10, type=int)
    args = parser.parse_args()

    modeldir = f'r{args.replace}_{args.model_name.split("/")[-1].replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir + f"_{args.bias_file.split('/')[-1].split('.')[0]}"
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Half precision to let the 20b model fit.
    torch_dtype = torch.float16 if "20b" in args.model_name else None

    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype)

    # Embedding
    subjects = json.load(open(args.subject_file))

    # Bias Dataset
    knowns = StereoSetDataset(mt.tokenizer, args.bias_file, args.model_name)

    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Automatic spherical gaussian
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt, subjects
            )
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level == "m":
            # Automatic multivariate gaussian
            noise_level = collect_embedding_gaussian(mt)
            print(f"Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            # Automatic d-distribution with d degrees of freedom
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])
    
    

    for knowledge in tqdm(knowns):
        ifskip = False
        for word in knowledge['subject']:
            if word not in knowledge['anti'] or word not in knowledge['stereo']:
                print(f"Skipping {knowledge['id']}")
                ifskip = True
                break
        if ifskip:
            continue
        known_id = knowledge["id"]

        # original difference: base_score
        if mt.iscausal:
            inp_anti, e_range_anti, blank_idxs_anti, inp_anti_origin = make_inputs(mt, prompts=[knowledge['anti']] * (args.samples + 1), labels=[knowledge['anti_mask']] * (args.samples + 1), subject=knowledge['subject'])
            inp_stereo, e_range_stereo, blank_idxs_stereo, inp_stereo_origin = make_inputs(mt, prompts=[knowledge['stereo']] * (args.samples + 1), labels=[knowledge['stereo_mask']] * (args.samples + 1), subject=knowledge['subject'])
            if (inp_anti==None and e_range_anti==None and blank_idxs_anti==None and inp_stereo_origin==None) or (inp_stereo==None and e_range_stereo==None and blank_idxs_stereo==None and inp_stereo_origin==None):
                continue
            if inp_anti["input_ids"].shape[1] != inp_stereo["input_ids"].shape[1]:
                continue
            with torch.no_grad():
                pred_anti = _logits(mt.model(**inp_anti))
                targ_anti = inp_anti["labels"]
                pred_stereo = _logits(mt.model(**inp_stereo))
                targ_stereo = inp_stereo["labels"]
                base_score = causal_difference(pred_anti, targ_anti, pred_stereo, targ_stereo)
                print(base_score)   # before interrupting
                # if base_score: print("Hello")
        else:
            inp_anti, e_range_anti, blank_idxs_anti, inp_anti_origin = make_inputs(mt, prompts=[knowledge['anti_mask']] * (args.samples + 1), labels=[knowledge['anti']] * (args.samples + 1), subject=knowledge['subject'])
            inp_stereo, e_range_stereo, blank_idxs_stereo, inp_stereo_origin = make_inputs(mt, prompts=[knowledge['stereo_mask']] * (args.samples + 1), labels=[knowledge['stereo']] * (args.samples + 1), subject=knowledge['subject'])
            if (inp_anti==None and e_range_anti==None and blank_idxs_anti==None and inp_anti_origin==None) or (inp_stereo==None and e_range_stereo==None and blank_idxs_stereo==None and  inp_stereo_origin==None):
                continue
            if inp_anti["input_ids"].shape[1] != inp_stereo["input_ids"].shape[1]:
                continue
            with torch.no_grad():
                pred_anti = _logits(mt.model(**inp_anti))
                targ_anti = inp_anti["labels"]
                pred_stereo = _logits(mt.model(**inp_stereo))
                targ_stereo = inp_stereo["labels"]
                base_score = mask_difference(pred_anti, targ_anti, pred_stereo, targ_stereo) 
        
        # difference after corrupting the embedding of bias attribute words, the lowest difference
        anti_outputs = trace_with_patch(
            model=mt.model,
            inp=inp_anti, 
            states_to_patch=[], 
            tokens_to_mixs=e_range_anti, # bias attribute words
            noise=noise_level, 
            uniform_noise=uniform_noise
        )

        stereo_outputs = trace_with_patch(
            model=mt.model,
            inp=inp_stereo, 
            states_to_patch=[], 
            tokens_to_mixs=e_range_stereo, 
            noise=noise_level, 
            uniform_noise=uniform_noise
        )
        # Outputs with corrupted the embedding of bias attribute words
        pred_anti = _logits(anti_outputs)
        targ_anti = inp_anti["labels"]
        pred_stereo = _logits(stereo_outputs)
        targ_stereo = inp_stereo["labels"]

        if mt.iscausal:
            # We report the difference of log probabilities for the whole sentence except the corrupted tokens.
            low_score = causal_difference(pred_anti[1:], targ_anti[1:], pred_stereo[1:], targ_stereo[1:])
        else:
            # We report difference log probabilities for the masked tokens.
            low_score = mask_difference(pred_anti[1:], targ_anti[1:], pred_stereo[1:], targ_stereo[1:])


        for kind in None, "mlp", "attn":
            if kind=="mlp" and not mt.iscausal:
                kind = "intermediate"
            print(f"Causal Tracing for {known_id} {kind} ==========================================================")
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.npz"
            if not os.path.isfile(filename):
                result = calculate_hidden_flow(
                    mt,
                    knowledge,
                    inp_anti,                       # context
                    inp_stereo,
                    e_range_anti,                   # bias attribute word
                    e_range_stereo,
                    blank_idxs_anti,                # bias term
                    blank_idxs_stereo,
                    inp_anti_origin,                # label for context
                    inp_stereo_origin,                    
                    noise=noise_level,
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                    kind=kind,
                )
                if not result:
                    print(f"Skipping {knowledge['id']}")
                    continue
                result["high_score"] = base_score               # before causal tracing, the original prob difference of bias term                                                                   
                result["low_score"] = low_score                 # after corrupting embedding, return the prob difference of bias term
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                numpy.savez(filename, **numpy_result)
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            # if not numpy_result["correct_prediction"]:
            #     tqdm.write(f"Skipping {knowledge['prompt']}")
            #     continue
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            pdfname = f'{pdf_dir}/{known_id}_{str("_".join(numpy_result["subject"])).strip()}_{kind_suffix}'
            plot_trace_heatmap(plot_result, savepdf_pre=pdfname, modelname=mt.model_name)


def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    tokens_to_mixs, # Range List of tokens to corrupt (begin, end); subject tokens range
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mixs specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            # corrupt all subjects
            if tokens_to_mixs is not None:
                for tokens_to_mix in tokens_to_mixs:
                    b, e = tokens_to_mix
                    noise_data = noise_fn(
                        torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                    ).to(x.device)
                    if replace:
                        x[1:, b:e] = noise_data
                    else:
                        x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers

    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return outputs_exp, all_traced

    return outputs_exp

def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mixs,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
):
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mixs is not None:
                for tokens_to_mix in tokens_to_mixs:
                    b, e = tokens_to_mix
                    x[1:, b:e] += noise * torch.from_numpy(
                        prng(x.shape[0] - 1, e - b, x.shape[2])
                    ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


def _logits(x):
    return x if not hasattr(x, "logits") else x.logits

def mask_difference(pred_anti, targ_anti, pred_stereo, targ_stereo):
    anti_soft = F.log_softmax(pred_anti, dim=-1)

    def get_edit(p,t):
        mask_idxs = (t != -100)
        template_word_idxs = [] # ids of [MASK] target tokens
        for i in range(len(t)): # seq_len
            if t[i] != -100:
                template_word_idxs.append(t[i])
                # anti_n_tokens += 1
        ele_probs = p[mask_idxs] # [MASK] probs, (mask_num, vocab_size)
        ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(pred_anti.device)).diag()   # [MASK] target tokens probs (mask_num,)
        return ele_probs.mean()
    
    anti_score = get_edit(anti_soft[0], targ_anti[0]).unsqueeze(0)
    for batch_idx, (p, t) in enumerate(zip(anti_soft, targ_anti)):
        if batch_idx==0:
            continue
        anti_score = torch.cat((anti_score, get_edit(p,t).unsqueeze(0)), dim=0)                                              # mean probs of [MASK] target tokens
    

    stereo_soft = F.log_softmax(pred_stereo, dim=-1)
    # stereo_n_tokens = 0
    stereo_score = get_edit(stereo_soft[0], targ_stereo[0]).unsqueeze(0)
    for batch_idx, (p, t) in enumerate(zip(stereo_soft, targ_stereo)):
        if batch_idx==0:
            continue
        stereo_score = torch.cat((stereo_score, get_edit(p,t).unsqueeze(0)), dim=0)

    return torch.abs(anti_score.mean() - stereo_score.mean())


def causal_difference(pred_anti, targ_anti, pred_stereo, targ_stereo):
    
    if pred_anti.dim() == 3 and pred_stereo.dim() == 3:  # Dealing with sequences
        pred_anti = pred_anti[:, :-1]  # Remove last prediction in sequence
        targ_anti = targ_anti[:, 1:]  # Shift to align predictions and targets
        pred_stereo = pred_stereo[:, :-1]
        targ_stereo = targ_stereo[:, 1:]
    
    def get_score(p,t):
            mask_idxs = (t != -100)
            template_word_idxs = []
            for i in range(len(t)):
                if t[i] != -100:
                    template_word_idxs.append(t[i])
            ele_probs = p.log_softmax(-1)[mask_idxs]
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(pred_anti.device)).diag()
            return ele_probs.mean()
    
    anti_score = get_score(pred_anti[0], targ_anti[0]).unsqueeze(0)
    for batch_idx, (p, t) in enumerate(zip(pred_anti, targ_anti)):
        if batch_idx==0:
            continue
        anti_score = torch.cat((anti_score, get_score(p, t).unsqueeze(0)), dim=0)   # (batch_size, )
          

    stereo_score = get_score(pred_stereo[0], targ_stereo[0]).unsqueeze(0)
    for batch_idx, (p, t) in enumerate(zip(pred_stereo, targ_stereo)):
        if batch_idx==0:
            continue
        stereo_score = torch.cat((stereo_score, get_score(p, t).unsqueeze(0)), dim=0)   # (batch_size, )
    
    return torch.abs(anti_score.mean() - stereo_score.mean())
    
    # return F.kl_div(input=anti_score, target=stereo_score, reduction="batchmean", log_target=False) + F.kl_div(input=F.softmax(stereo_score, dim=-1), target=anti_score, reduction="batchmean", log_target=False)
   

def calculate_hidden_flow(
    mt,
    knowledge,
    inp_anti, 
    inp_stereo,
    e_range_anti, 
    e_range_stereo,
    blank_idxs_anti, 
    blank_idxs_stereo,
    inp_anti_origin,
    inp_stereo_origin,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    
    
    # difference after corrupting embedding and restoring
    
    if not kind:
        differences = trace_important_states(
            mt,
            inp_anti, inp_stereo, 
            e_range_anti, e_range_stereo,
            blank_idxs_anti, blank_idxs_stereo,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            # token_range=token_range,
        )
    else:
        differences = trace_important_window(
            mt,
            inp_anti, inp_stereo,
            e_range_anti, e_range_stereo,
            blank_idxs_anti, blank_idxs_stereo,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            # token_range=token_range,
        )
    differences = differences.detach().cpu()                            #(seq_len, num_layers)
    return dict(
        scores=differences,                                             
        anti_input_ids=inp_anti["input_ids"][0],                               # input_ids of the prompt
        stereo_input_ids=inp_stereo['input_ids'][0],
        input_tokens_anti=decode_tokens(mt.tokenizer, inp_anti["input_ids"][0]) if mt.iscausal else decode_tokens(mt.tokenizer, inp_anti_origin['input_ids'][0]),  # tokens of the prompt
        input_tokens_stereo=decode_tokens(mt.tokenizer, inp_stereo["input_ids"][0]) if mt.iscausal else decode_tokens(mt.tokenizer, inp_stereo_origin['input_ids'][0]),
        corrupt_range_anti=e_range_anti,
        corrupt_range_stereo=e_range_stereo,
        blank_idxs_anti=blank_idxs_anti,
        blank_idxs_stereo=blank_idxs_stereo,           # bias term idx range in input_ids
        subject=knowledge['subject'],                  
        window=window,
        # correct_prediction=True,
        kind=kind or "",
    )


def trace_important_states(
    mt,
    inp_anti, inp_stereo,
    e_range_anti, e_range_stereo,
    blank_idxs_anti, blank_idxs_stereo,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    # token_range=None,
):
    ntoks_anti = inp_anti["input_ids"].shape[1]
    ntoks_stereo = inp_stereo['input_ids'].shape[1]
    assert ntoks_anti == ntoks_stereo

    
    # if token_range is None:
    # token_range_anti = list(set(range(ntoks_anti)) - set(blank_idxs_anti))
    # token_range_stereo = list(set(range(ntoks_stereo)) - set(blank_idxs_stereo))
    # assert len(token_range_stereo) == len(token_range_anti), "After remove blank tokens, anti and stereo should have the same length"
    
    table = [] # (num_layers, seq_len)

    for tnum in range(ntoks_anti):
        row = []
        for layer in range(mt.num_layers):
            anti_outputs = trace_with_patch(
                model=mt.model,
                inp=inp_anti,
                states_to_patch=[(tnum, layername(mt.model, layer))],
                tokens_to_mixs=e_range_anti,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            stereo_outputs = trace_with_patch(
                model=mt.model,
                inp=inp_stereo,
                states_to_patch=[(tnum, layername(mt.model, layer))],
                tokens_to_mixs=e_range_stereo,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            pred_anti = _logits(anti_outputs)
            targ_anti = inp_anti["labels"]
            pred_stereo = _logits(stereo_outputs)
            targ_stereo = inp_stereo["labels"]

            if mt.iscausal:
                # We report the difference of softmax probabilities for the whole sentence except the corrupted tokens.
                r = causal_difference(pred_anti[1:], targ_anti[1:], pred_stereo[1:], targ_stereo[1:])
            else:
                # We report difference softmax probabilities for the masked tokens.
                r = mask_difference(pred_anti[1:], targ_anti[1:], pred_stereo[1:], targ_stereo[1:])
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    mt,
    inp_anti, inp_stereo,
    e_range_anti, e_range_stereo,
    blank_idxs_anti, blank_idxs_stereo,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    # token_range=None,
):
    ntoks_anti = inp_anti["input_ids"].shape[1]
    ntoks_stereo = inp_stereo['input_ids'].shape[1]
    assert ntoks_anti == ntoks_stereo

     # if token_range is None:
    # token_range_anti = list(set(range(ntoks_anti)) - set(blank_idxs_anti))
    # token_range_stereo = list(set(range(ntoks_stereo)) - set(blank_idxs_stereo))
    # assert len(token_range_stereo) == len(token_range_anti), "After remove blank tokens, anti and stereo should have the same length"
    
    table = [] # (num_layers, seq_len)
    for tnum in range(ntoks_anti):
        row = []
        for layer in range(mt.num_layers):
            layerlist_anti = [
                (tnum, layername(mt.model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(mt.num_layers, layer - (-window // 2))
                )
            ]
            anti_outputs = trace_with_patch(
                mt.model,
                inp=inp_anti,
                states_to_patch=layerlist_anti,
                tokens_to_mixs=e_range_anti,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )

            layerlist_stereo = [
                (tnum, layername(mt.model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(mt.num_layers, layer - (-window // 2))
                )
            ] 
            stereo_outputs = trace_with_patch(
                mt.model,
                inp=inp_stereo,
                states_to_patch=layerlist_stereo,
                tokens_to_mixs=e_range_stereo,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )

            pred_anti = _logits(anti_outputs)
            targ_anti = inp_anti["labels"]
            pred_stereo = _logits(stereo_outputs)
            targ_stereo = inp_stereo["labels"]
            if mt.iscausal:
                # We report the difference of softmax probabilities for the whole sentence except the corrupted tokens.
                r = causal_difference(pred_anti[1:], targ_anti[1:], pred_stereo[1:], targ_stereo[1:])
            else:
                # We report difference softmax probabilities for the masked tokens.
                r = mask_difference(pred_anti[1:], targ_anti[1:], pred_stereo[1:], targ_stereo[1:])
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            if "llama" in model_name.lower():
                tokenizer = LlamaTokenizer.from_pretrained(model_name)
            elif model_name=="gpt2-medium":
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
        if model is None:
            assert model_name is not None
            if "llama" in model_name.lower():
                model = LlamaForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
                )
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                model.model.embed_tokens.weight.data[-1] = model.model.embed_tokens.weight.data.mean(0)
            elif "gpt" in model_name.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
                )
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)
            else:
                model = AutoModelForMaskedLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
                )
            nethook.set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model|bert|roberta)\.(h|layers|encoder.layer)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)
        if "gpt" in model_name.lower() or "llama" in model_name.lower():
            self.iscausal = True
        else:
            self.iscausal = False
        self.model_name = model_name

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
    if hasattr(model, "bert"):
        if kind == "embed":
            return "bert.embeddings.word_embeddings"
        if kind == "attn":
            kind = "attention"
        return f'bert.encoder.layer.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "roberta"):
        if kind == "embed":
            return "roberta.embeddings.word_embeddings"
        if kind == "attn":
            kind = "attention"
        return f'roberta.encoder.layer.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "model"): # llama
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            return f'model.layers.{num}.self_attn'
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()



def plot_trace_heatmap(result, savepdf_pre=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["subject"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels_anti = list(result["input_tokens_anti"])
    labels_stereo = list(result['input_tokens_stereo'])

    # anti
    for e_range in result['corrupt_range_anti']:
        for i in range(*e_range):
            labels_anti[i] = labels_anti[i] + "*"
    labels_anti[result['blank_idxs_anti'][0]] = "[" + labels_anti[result['blank_idxs_anti'][0]].strip()
    labels_anti[result['blank_idxs_anti'][1]-1] = labels_anti[result['blank_idxs_anti'][1]-1].strip() + "]"

    savepdf = savepdf_pre+"_anti.pdf"
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "intermediate": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels_anti)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after the corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if (kind == "mlp" or kind == "intermediate") else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            
            cb.ax.set_title(f"Corrupt: {' '.join(answer)}", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
    #stereo
    for e_range in result['corrupt_range_stereo']:
        for i in range(*e_range):
            labels_stereo[i] = labels_stereo[i] + "*"
    labels_stereo[result['blank_idxs_stereo'][0]] = "[" + labels_stereo[result['blank_idxs_stereo'][0]].strip()
    labels_stereo[result['blank_idxs_stereo'][1]-1] = labels_stereo[result['blank_idxs_stereo'][1]-1].strip() + "]"
    
    
    savepdf = savepdf_pre+"_stereo.pdf"
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "intermediate": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels_stereo)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if (kind == "mlp" or kind == "intermediate") else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.

            cb.ax.set_title(f"Corrupt: {' '.join(answer)}", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    


# def plot_all_flow(mt, prompt, subject=None):
#     for kind in ["mlp", "attn", None]:
#         plot_hidden_flow(mt, prompt, subject, kind=kind)


# Utilities for dealing with tokens
def make_inputs(mt, prompts, labels, subject=None, device="cuda"):    
    if "gpt" in mt.model_name.lower():
        prompts = [mt.tokenizer.bos_token + p for p in prompts]
        labels = [mt.tokenizer.bos_token + p for p in labels]   
    
    inputs = mt.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            # truncation=True
    )
    inputslabels = mt.tokenizer(
            labels,
            padding=True,
            return_tensors="pt",
            # truncation=True
    )
    if inputs['input_ids'].size()[1] != inputslabels['input_ids'].size()[1]:
        return None, None, None, None

    # assert inputs['input_ids'].size()[1] == inputslabels['input_ids'].size()[1], "inputs and labels should have the same length"

    if mt.iscausal:  # prompts with original sentences, labels with unk_token
        subject_range = []
        for subj in subject:
            subject_range.append(find_token_range(mt.tokenizer, inputs["input_ids"][0], subj))

        inputs['labels'] = copy.deepcopy(inputs['input_ids'])

        for idx in range(len(inputs['labels'])):
            inputs['labels'][idx] = torch.where(inputs['input_ids'][idx] != mt.tokenizer.pad_token_id, inputs['input_ids'][idx], -100)  # ignore pad_tokens
            # for (b,e) in subject_range:             # ignore subjects
            #     inputs['labels'][idx][b:e] = -100   
            inputs['labels'][idx][0] = -100         # ignore bos_token
        blank_token_idxs = find_token_range(mt.tokenizer, inputslabels['input_ids'][0][1:], mt.tokenizer.unk_token)
        blank_token_idxs = (blank_token_idxs[0]+1, blank_token_idxs[1]+1)
    else:
        subject_range = []
        for subj in subject:
            subject_range.append(find_token_range(mt.tokenizer, inputslabels["input_ids"][0], subj))
        inputs['labels'] = copy.deepcopy(inputslabels['input_ids'])
        for idx in range(len(inputs['labels'])):    # prompts with [MASK], labels with original sentences
            inputs['labels'][idx] = torch.where(inputs['input_ids'][idx] == mt.tokenizer.mask_token_id, inputslabels['input_ids'][idx], -100)
        
        blank_token_idxs = find_token_range(mt.tokenizer, inputs['input_ids'][0], mt.tokenizer.mask_token)
    
    return inputs.to(device), subject_range, blank_token_idxs, inputslabels


    # token_lists = [mt.tokenizer.encode(p) for p in prompts]
    # maxlen = max(len(t) for t in token_lists)
    # if "[PAD]" in mt.tokenizer.all_special_tokens:
    #     pad_id = mt.tokenizer.all_special_ids[mt.tokenizer.all_special_tokens.index("[PAD]")]
    # else:
    #     pad_id = 0
    # input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    # attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]                  # left pad?
    # return dict(
    #     input_ids=torch.tensor(input_ids).to(device),
    #     #    position_ids=torch.tensor(position_ids).to(device),
    #     attention_mask=torch.tensor(attention_mask).to(device),
    # )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substrings):    # find substring in token_array, return [start, end)

    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substrings)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substrings):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):         
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)        # the next token probability (batch_size, vocab_size)
    p, preds = torch.max(probs, dim=1)              # p is probability, preds is the mex token id of the higest probability for each instance
    return preds, p


def collect_embedding_std(mt, subjects, device="cuda"):
    alldata = []
    with torch.no_grad():
        for s in tqdm(subjects, desc="Collect Embeddings"):
            inp = mt.tokenizer(
                    [s],
                    padding=True,
                    return_tensors="pt"
            )

            inp.to(device)
            with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
                mt.model(**inp)
                alldata.append(t.output[0])     # t.output (batch_size, seq_len, emb_size)
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()  # the standard deviation over embeddings of all subjects
    return noise_level


def get_embedding_cov(mt):
    model = mt.model
    tokenizer = mt.tokenizer

    def get_ds():
        ds_name = "wikitext"
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = 100  # Hack due to missing setting in GPT2-NeoX.
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    ds = get_ds()
    sample_size = 1000
    batch_size = 5
    filename = None
    batch_tokens = 100

    progress = lambda x, **k: x

    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                del batch["position_ids"]
                with nethook.Trace(model, layername(mt.model, 0, "embed")) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()


def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer


def collect_embedding_gaussian(mt):
    m, c = get_embedding_cov(mt)
    return make_generator_transform(m, c)


def collect_embedding_tdist(mt, degree=3):
    # We will sample sqrt(degree / u) * sample, where u is from the chi2[degree] dist.
    # And this will give us variance is (degree / degree - 2) * cov.
    # Therefore if we want to match the sample variance, we should
    # reduce cov by a factor of (degree - 2) / degree.
    # In other words we should be sampling sqrt(degree - 2 / u) * sample.
    u_sample = torch.from_numpy(
        numpy.random.RandomState(2).chisquare(df=degree, size=1000)
    )
    fixed_sample = ((degree - 2) / u_sample).sqrt()
    mvg = collect_embedding_gaussian(mt)

    def normal_to_student(x):
        gauss = mvg(x)
        size = gauss.shape[:-1].numel()
        factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
        student = factor * gauss
        return student

    return normal_to_student


if __name__ == "__main__":
    main()
