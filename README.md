This repository contains the source code submitted by LMU Munich to the WMT 2020 Unsupervised MT Shared Task. For a detailed description, check our [paper](http://statmt.org/wmt20/pdf/2020.wmt-1.128.pdf). 

Our system ranked **first** in both translation directions (German -> Sorbian, Sorbian->German). This code base is largely based on [MASS](https://github.com/microsoft/MASS/) and [RE-LM](https://github.com/alexandra-chron/relm_unmt/).


# Introduction

 The target of the task was to translate between German and Upper Sorbian (minority language of Eastern Germany, similar to Czech). Our system is based on a **combination of Unsupervised Neural MT and Unsupervised Statistical MT**.
 
- For the **Neural MT** part, we use [MASS](https://www.microsoft.com/en-us/research/uploads/prod/2019/06/MASS-paper-updated-002.pdf). However, instead of pretraining on German and Sorbian, we pretrain only on German. Upon convergence, we extend the vocabulary of the pretrained model and fine-tune it to Sorbian and German. This follows [RE-LM](https://www.aclweb.org/anthology/2020.emnlp-main.214.pdf), a competitive method for low-resource unsupervised NMT. Then, we train for NMT in an unsupervised way (online backtranslation). 
 
- For the **Statistical MT** part, we use [monoses](https://github.com/artetxem/monoses). Specifically, we map [fastText](https://github.com/facebookresearch/fastText) embeddings using VecMap with identical pairs. Then, we backtranslate and get a pseudo-parallel corpus for both directions.  We train our NMT system using online BT *and* a supervised loss on the pseudo-parallel corpus from USMT. 
 
 Also useful:
 
 - Sampling when doing the prediction during online BT instead of greedy decoding. See flags ``--sampling_frequency``, ``--sample_temperature`` in the code. 

- Oversampling the Sorbian corpus using BPE-Dropout. We preprocess data using [subword-nmt](https://github.com/rsennrich/subword-nmt#advanced-features) with the flag `--dropout 0.1`. 


Our proposed pipeline:

<img src="https://github.com/alexandra-chron/umt-lmu-wmt2020/blob/main/system_overview.png" width="800">

Right arrows indicate transfer of weights. 

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

#### Data
You can download all the German Newscrawl data, all the Sorbian monolingual data, and the evaluation/test sets from the [WMT](http://www.statmt.org/wmt20/unsup_and_very_low_res/) official website.


# Training a baseline UNMT model and adding pseudo-parallel data from USMT

### 1. Pretrain a German encoder-decoder model with attention using the MASS pretraining objective

To preprocess your data using BPE tokenization, make sure you have placed them in ``./data/de-wmt``. Then run:

``` ./get_data_mass_pretraining.sh --src de ```

Then, train the German MASS model:

```
python3 train.py --exp_name de_mass --dump_path './models' --data_path './data/de-wmt' --lgs de --mass_steps de --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true --tokens_per_batch 2000 --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --epoch_size 200000 --max_epoch 100000 --word_mass '0.5' --min_len 5 
```

### 2. Fine-tune the MASS model using Sorbian and German

Before this step, you need to extend the vocabulary to accound for the new, Sorbian BPE vocabulary items. Specifically, the embedding layer (and the output layer) of the MASS model need to be increased by the amount of new items added to the existing vocabulary for this step. To do that, use the following command:

``./get_data_and_preprocess.sh --src de --tgt hsb``

In the directory ./data/de-hsb-wmt/, a file named vocab.hsb-de-ext-by-$NUMBER has been created. This number indicates by how many items we need to extend the initial vocabulary, and consequently the embedding and linear layer, to account for the Hsb language.

You will need to give this value to the ``--increase_vocab_by`` argument so that you successfully run the fine-tuning step of MASS.

Then, fine-tune the model: 

```
python3 train.py --exp_name de_mass_ft_hsb --dump_path './models' --data_path './data/de-hsb-wmt/' --lgs 'de-hsb' --mass_steps 'de,hsb' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true --tokens_per_batch 2000 --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --epoch_size 50000 --max_epoch 100000 --word_mass '0.5' --min_len 5 --reload_model './models/de_mass/3w8dqrykpd/checkpoint.pth' --increase_vocab_for_lang de --increase_vocab_from_lang hsb --increase_vocab_by $NUMBER
```
 
### 3. Train the fine-tuned MASS for UNMT, with online BT (+ sampling)

```
python3 train.py --exp_name 'unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5' --dump_path './models' --data_path './data/de-hsb-wmt/'  --lgs 'de-hsb' --bt_steps 'de-hsb-de,hsb-de-hsb' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true --tokens_per_batch 1000 --batch_size 32 --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --epoch_size 50000 --max_epoch 100000 --eval_bleu true --sample_temperature '0.95' --reload_model './models/de_mass_ft_hsb/8fark50w1p/checkpoint.pth,./models/de_mass_ft_hsb/8fark50w1p/checkpoint.pth' --increase_vocab_for_lang de --increase_vocab_from_lang hsb --sampling_frequency '0.5'
```

### 4. Fine-tune the UNMT model, using both a BT loss on the monolingual data and a supervised loss on the pseudo-parallel data from USMT

Assuming you have created pseudo-parallel data from USMT and placed them in ``./data/de-hsb-wmt`` in the following form:


- train.hsb-de.{de, hsb}: original de monolingual data, hsb backtranslations

- train.de-hsb.{de, hsb}: original hsb monolingual data, de backtranslations

This will be used as a pseudo-parallel corpus (``--mt_steps`` flag):


``` 
python3 train.py --exp_name 'unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5_ft_smt_both_dir' --dump_path './models' --data_path './data/de-hsb-wmt' --lgs 'de-hsb' --ae_steps 'de,hsb' --bt_steps 'de-hsb-de,hsb-de-hsb' --mt_steps 'de-hsb,hsb-de' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true --tokens_per_batch 1000 --batch_size 32 --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --epoch_size 50000 --max_epoch 100000 --eval_bleu true --increase_vocab_for_lang de --increase_vocab_from_lang hsb --reload_model './models/unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5/fsp0smjzgu/checkpoint.pth,./models/unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5/fsp0smjzgu/checkpoint.pth' --sampling_frequency '0.5' --sample_temperature '0.95' --load_diff_mt_direction_data true 
```

### 5. Use the trained model (from 4) to backtranslate data in both directions (inference)

It is better to pick a subset of train.de, as it will probably be very large (we downloaded 327M sentences from NewsCrawl).

```
ln -s ./de-hsb-wmt/train.de to ./temp/train.hsb-de.de
ln -s ./de-hsb-wmt/train.hsb to ./temp/train.de-hsb.hsb
```
Then, run the NMT model:
```
python3 translate.py --src_lang de --tgt_lang hsb --model_path ./models/unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5_ft_smt_both_dir/tca9s0sr08/checkpoint.pth --exp_name translate_de_hsb_750k --dump_path './models' --output_path  ./data/temp/train.hsb-de.hsb --batch_size 64 --input_path ./data/temp/train.hsb-de.de --beam 5
```
- train.hsb-de.de will contain the *original* German data

- train.hsb-de.hsb will contain the *backtranslated* Sorbian data

This will be used as a pseudo-parallel corpus in the next step. 

```
python3 translate.py --src_lang hsb --tgt_lang de --model_path ./models/unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5_ft_smt_both_dir/tca9s0sr08/checkpoint.pth --exp_name translate_hsb_de_750k --dump_path './models' --output_path  ./data/temp/train.hsb-de.de --batch_size 64 --input_path ./data/temp/train.hsb-de.hsb --beam 5
```

Accordingly,

- train.de-hsb.de will contain the *backtranslated* German data

- train.de-hsb.hsb will contain the *original* Sorbian data

After you store the USMT pseudo-parallel corpus (``./data/de-hsb-wmt.train.{hsb-de,de-hsb}.{de,hsb}`` in a different directory, put the ``./data/temp/train.{hsb-de,de-hsb}.{hsb,de}`` files in the ``./data/de-hsb-wmt`` directory, in order to use them in step 6. 

### 6. Use the trained model (step 4) + the pseudo-parallel data from 5 to further train an NMT model


``` 
python3 train.py --exp_name 'unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5_ft_smt_both_dir' --dump_path './models' --data_path './data/de-hsb-wmt/'  --lgs 'de-hsb' --ae_steps 'de,hsb' --bt_steps 'de-hsb-de,hsb-de-hsb' --mt_steps 'de-hsb,hsb-de' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true --tokens_per_batch 1000 --batch_size 32 --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --epoch_size 50000 --max_epoch 100000 --eval_bleu true --increase_vocab_for_lang de --increase_vocab_from_lang hsb --reload_model './models/unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5/fsp0smjzgu/checkpoint.pth,./models/unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5/fsp0smjzgu/checkpoint.pth' --sampling_frequency '0.5' --sample_temperature '0.95' --load_diff_mt_direction_data true 
```

### 7. BPE-dropout on Hsb corpus and fine-tuning the NMT model

After you oversample the Hsb corpus, apply BPE-dropout to it using ``apply-bpe`` from [subword-nmt](https://github.com/rsennrich/subword-nmt#advanced-features) with the flag `--dropout 0.1`. 

Then, place it in a directory ``./data/de-hsb-wmt-bpe-dropout``, together with the De data and run

```
python3 train.py --exp_name cont_from_best_unmt_bpe_drop --dump_path './models' --data_path './data/de-hsb-wmt-bpe-dropout' --lgs 'de-hsb' --bt_steps 'de-hsb-de,hsb-de-hsb'  --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true --tokens_per_batch 1000 --batch_size 32 --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --epoch_size 50000 --max_epoch 100000 --eval_bleu true --increase_vocab_for_lang de --increase_vocab_from_lang hsb --reload_model 'unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5_ft_smt_both_dir/saa386ltp2/checkpoint.pth,unsup_nmt_de_mass_ft_hsb_ft_nmt_sampling_th-0.95_spl-0.5_ft_smt_both_dir/saa386ltp2/checkpoint.pth' --sampling_frequency '0.5' --sample_temperature '0.95'
```


#  Reference

If you use our work, please cite: 

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




