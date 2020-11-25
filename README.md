This repository contains the source code submitted by LMU Munich to the WMT 2020 Unsupervised MT Shared Task. For a detailed description, check our [paper](https://arxiv.org/abs/2010.13192). 

Our system ranked **first** in both translation directions (German -> Sorbian, Sorbian->German). This code base is largely based on [MASS](https://github.com/microsoft/MASS/) and [RE-LM](https://github.com/alexandra-chron/relm_unmt/).


# Introduction

 The target of the task was to translate between German and Upper Sorbian (minority language of Eastern Germany, similar to Czech). Our system is based on a **combination of Unsupervised Neural MT and Unsupervised Statistical MT**.
 
- For the **Neural MT** part, we use [MASS](https://www.microsoft.com/en-us/research/uploads/prod/2019/06/MASS-paper-updated-002.pdf). However, instead of pretraining on German and Sorbian, we pretrain only on German. Upon convergence, we extend the vocabulary of the pretrained model and fine-tune it to Sorbian and German. This follows [RE-LM](https://www.aclweb.org/anthology/2020.emnlp-main.214.pdf). Then, we train for NMT in an unsupervised way (online backtranslation). 
 
- For the **Statistical MT** part, we use [monoses](https://github.com/artetxem/monoses). Specifically, we map [fastText](https://github.com/facebookresearch/fastText) embeddings using VecMap with identical pairs. Then, we backtranslate and get a pseudo-parallel corpus for both directions.  We train our NMT system using online BT *and* an supervised loss on the pseudo-parallel corpus from USMT. 
 
 Also useful:
 
 - Sampling when doing the prediction during online BT instead of greedy decoding. See flags ``--sampling_frequency``, ``--sample_temperature`` in the code. 

- Oversampling the Sorbian corpus using BPE-Dropout. We preprocess data using [subword-nmt](https://github.com/rsennrich/subword-nmt#advanced-features) with the flag `--dropout 0.1`. 


Our proposed pipeline:

<img src="https://github.com/alexandra-chron/umt-lmu-wmt2020/blob/main/system_overview.png" width="800">

# Prerequisites 

#### Dependencies

- Python 3.6.9
- [NumPy](http://www.numpy.org/) (tested on version 1.15.4)
- [PyTorch](http://pytorch.org/) (tested on version 1.2.0)
- [Apex](https://github.com/NVIDIA/apex#quick-start) (for fp16 training)

#### Install Requirements 
**Create Environment (Optional):**  Ideally, you should create a conda environment for the project.

```
conda create -n wmt python=3.6.9
conda activate wmt
```

Install PyTorch ```1.2.0``` with the desired cuda version to use the GPU:

``` conda install pytorch==1.2.0 torchvision -c pytorch```

Clone the project:

```
git clone https://github.com/alexandra-chron/umt-lmu-wmt2020.git

cd umt-lmu-wmt2020
```

Then install the rest of the requirements:

```
pip install -r ./requirements.txt
```


To [train with multiple GPUs](https://github.com/facebookresearch/XLM#how-can-i-run-experiments-on-multiple-gpus) use:
```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
```


# Training a baseline UNMT model and adding pseudo-parallel data from USMT

### 1. Pretrain a German encoder-decoder model with attention using the MASS pretraining objective.

After preprocessing NewsCrawl German data using BPE tokenization (we used [fastBPE](https://github.com/glample/fastBPE)) with 32K merge operations and placing the data in ``./data/de-wmt``: 

```
python3 train.py --exp_name de_mass --dump_path './models' --data_path './data/de-wmt' --lgs de --mass_steps de --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true --tokens_per_batch 2000 --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --epoch_size 200000 --max_epoch 100000 --word_mass '0.5' --min_len 5 
```

### 2. Fine-tune the MASS model using Sorbian and German.

```
python3 train.py --exp_name de_mass_ft_hsb --dump_path './models' --data_path './data/de-sorbian-wmt/' --lgs 'de-hsb' --mass_steps 'de,hsb' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true --tokens_per_batch 2000 --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --epoch_size 50000 --max_epoch 100000 --word_mass '0.5' --min_len 5 --reload_model './models/de_mass/3w8dqrykpd/checkpoint.pth' --increase_vocab_for_lang de --increase_vocab_from_lang hsb --increase_vocab_by 15634
```
 
### 3. Train the resulting fine-tuned MASS model for UNMT, using online BT loss (+ sampling).

```
python3 train.py --exp_name 'unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5' --dump_path './models' --data_path './data/de-sorbian-wmt/' --lgs 'de-hsb' --bt_steps 'de-hsb-de,hsb-de-hsb' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true --tokens_per_batch 1000 --batch_size 32 --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --epoch_size 50000 --max_epoch 100000 --eval_bleu true --sample_temperature '0.95' --reload_model './models/de_mass_ft_hsb_ft_nmt/8fark50w1p/checkpoint.pth,./models/de_mass_ft_hsb_ft_nmt/8fark50w1p/checkpoint.pth' --increase_vocab_for_lang de --increase_vocab_from_lang hsb --sampling_frequency '0.5'
```

### 4. Further train the UNMT model, using both a BT loss on the monolingual data and a supervised loss on the pseudo-parallel data from USMT.

``` 
python3 train.py --exp_name 'unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5_ft_smt_both_dir' --dump_path './models' --data_path './data/de-sorbian-wmt' --lgs 'de-hsb' --ae_steps 'de,hsb' --bt_steps 'de-hsb-de,hsb-de-hsb' --mt_steps 'de-hsb,hsb-de' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true --tokens_per_batch 1000 --batch_size 32 --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --epoch_size 50000 --max_epoch 100000 --eval_bleu true --increase_vocab_for_lang de --increase_vocab_from_lang hsb --reload_model './models/unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5/fsp0smjzgu/checkpoint.pth,./models/unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5/fsp0smjzgu/checkpoint.pth' --sampling_frequency '0.5' --sample_temperature '0.95' --load_diff_mt_direction_data true 
```

### 5. Use the trained model (from 4) to backtranslate monolingual data in both directions (inference on the trained NMT model).

```
python3 translate.py --src_lang de --tgt_lang hsb --model_path ./models/unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5_ft_smt_both_dir/tca9s0sr08/checkpoint.pth --exp_name translate_de_hsb_750k --dump_path './models' --output_path  ./data/de-wmt/train.hsb-de.hsb --batch_size 64 --input_path ./data/de-wmt/train.hsb-de.de --beam 5
```
- train.hsb-de.de will contain the original German data

- train.hsb-de.hsb will contain the backtranslated Sorbian data

This will be used as a *pseudo-parallel corpus*

```
python3 translate.py --src_lang hsb --tgt_lang de --model_path ./models/unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5_ft_smt_both_dir/tca9s0sr08/checkpoint.pth --exp_name translate_hsb_de_750k --dump_path './models' --output_path  ./data/translate_hsb_de_750k.de --batch_size 64 --input_path ./data/de-wmt/train.hsb-de.hsb --beam 5
```


#### Reference

```
@InProceedings{chronopoulou-EtAl:2020:WMT,
  author    = {Chronopoulou, Alexandra  and  Stojanovski, Dario  and  Hangya, Viktor  and  Fraser, Alexander},
  title     = {{T}he {LMU} {M}unich {S}ystem for the {WMT} 2020 {U}nsupervised {M}achine {T}ranslation {S}hared {T}ask},
  booktitle      = {Proceedings of the Fifth Conference on Machine Translation},
  month          = {November},
  year           = {2020},
  address        = {Online},
  publisher      = {Association for Computational Linguistics},
  pages     = {1082--1089},
  abstract  = {This paper describes the submission of LMU Munich to the WMT 2020 unsupervised shared task, in two language directions, Germanâ†”Upper Sorbian. Our core unsupervised neural machine translation (UNMT) system follows the strategy of Chronopoulou et al. (2020), using a monolingual pretrained language generation model (on German) and fine-tuning it on both German and Upper Sorbian, before initializing a UNMT model, which is trained with online backtranslation. Pseudo-parallel data obtained from an unsupervised statistical machine translation (USMT) system is used to fine-tune the UNMT model. We also apply BPE-Dropout to the low resource (Upper Sorbian) data to obtain a more robust system. We additionally experiment with residual adapters and find them useful in the Upper Sorbianâ†’German direction. We explore sampling during backtranslation and curriculum learning to use SMT translations in a more principled way. Finally, we ensemble our best-performing systems and reach a BLEU score of 32.4 on Germanâ†’Upper Sorbian and 35.2 on Upper Sorbianâ†’German.},
  url       = {https://www.aclweb.org/anthology/2020.wmt-1.128}
}

```




