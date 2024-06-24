# BiasEdit: Debiasing Stereotyped Language Models via Model Editing


<p align="center">
  <a href="">📃 Paper</a> 
  <a href="https://github.com/zjunlp/BiasEdit">💻 Code</a> 
  <a href="">🌏 Web</a> 
</p>



<div align=center><img src="fig/BiasEdit_fig1.gif" width="70%"/></div>


**BiasEdit** is an efficient *model editing* method to eliminate stereotyped bias from language models with small editor networks, including a *debiasing loss* to guide edits on partial parameters and a *remaining loss* to maintain the language modeling abilities during editing. Experimental results show BiasEdit' excellent performance on debiasing, modeling ability preservation, and robustness of gender reverse and semantic generality.

## 🆕 News
- [Feb 2024] We released the [paper]() and the refined code.
- [Dec 2023] Our [idea](https://www.winlp.org/winlp-2023-workshop/accepted-papers/) was accepted by WiNLP 2023 in EMNLP 2023!
- [Nov 2023] We released the code.

## 📌 Table of Contents

- [🛠️ Setup](#1)
- [💻 BiasEdit](#2)
    - [⌚️ Training Editor Networks](#2.1)
    - [🚀 Debiasing with Editor Networks](#2.2)
- [👀 Bias Tracing](#3)
- [📝 Citation](#4)
- [✨ Acknowledgements](#5)

<h2 id="1">🛠️ Setup</h2>

This codebase uses Python 3.9.18. Other versions may work as well.

Create an environment
and install the dependencies:

    $ conda create -n biasedit python=3.9
    $ conda activate biasedit
    (biasedit) $ pip install -r requirements.txt


<h2 id="2">💻 BiasEdit</h2>
<div align=center><img src="fig/BiasEdit_fig2.png" width="80%"/></div>

With [StereoSet](https://aclanthology.org/2021.acl-long.416/), editor networks are trained to generate parameter shifts for debiasing at first. Then, the trained editor networks are used to conduct edits on language models and produce an unbiased model.

<h3 id="2.1">⌚️ Training Editor Networks</h3>

- Formatted datasets with [train](./data/stereoset/train.json)/[dev](./data/stereoset/dev.json)/test (`gender_test.json`, `race_test.json`, `religion_test.json`) splits are in [data/stereoset](./data/stereoset). 
- Configurations are in [config](./config). Partial parameters to be edited are presented in [editor](./config/editor). The configurations, like weights to be edited, are in [model](config/model).
- Experimental scripts are in [scripts](./scripts). All hyper-parameters are in the scripts.
- For the ablation study on the remaining loss, set `editor.loc_coef=0`.
- Metrics can be found in the training log.

For example, we use the following command to train the editor networks for Gemma-2B:


```bash
 (biasedit) $ bash scripts/gemma_last2.sh
```


<h3 id="2.2">🚀 Debiasing with Editor Networks</h3>

- Set `eval_only=True`
- Set `data.valid_path` as the path of the test set
- Metrics can be found at the end of the debiasing log, like "Test ------- XXX".
- For testing the robustness of gender reverse, set `data.valid_path` as `data/stereoset/gender_test_reverse.json`.
- For testing the semantic generality, set `data.valid_path` as `data/stereoset/xxx_test_syn.json`, where `xxx` is chosen from [gender, race, religion].

For example,
```bash
 (biasedit) $ bash scripts/gpt2m_last123_gender_reverse.sh
```


<h2 id="3">👀 Bias Tracing</h2>

Enter [bias_tracing](./bias_tracing).


<h2 id="4">📝 Citation</h2>

If this code or paper was useful, please consider using the following citation:

    @article{xin24BiasEdit,
        title={BiasEdit: Debiasing Stereotyped Language Models via Model Editing},
        author={Xin Xu, Wei Xu, Ningyu Zhang},
        year={2024},
        url={https://github.com/zjunlp/BiasEdit}
    }

<h2 id="5">✨ Acknowledgements</h5>

- Thanks for the original code from [MALMEN](https://github.com/ChenmienTan/malmen) and [ROME](https://github.com/kmeng01/rome/blob/main/experiments/causal_trace.py).
- Thanks for StereoSet and all the baselines from [bias-bench](https://github.com/McGill-NLP/bias-bench).
- For more model editing methods, please try [EasyEdit](https://github.com/zjunlp/EasyEdit).
