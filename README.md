# System Submission from LMU Munich to WMT 2020 Unsupervised Machine Translation Shared Task

This repository contains the source code submitted by LMU Munich to the WMT 2020 Unsupervised MT Shared Task. [(Paper link)](https://arxiv.org/abs/2010.13192). The target of the task was to translate between German and Upper Sorbian (minority language of Eastern Germany, similar to Czech).

LMU Munich ranked first in both translation directions (German -> Sorbian, Sorbian->German). 

Our system is based on a combination of Unsupervised Neural MT and Unsupervised Statistical MT. 

- For the Neural MT part, we use [MASS](https://www.microsoft.com/en-us/research/uploads/prod/2019/06/MASS-paper-updated-002.pdf). However, instead of pretraining on German and Sorbian, we pretrain only on German. Upon convergence, we extend the vocabulary of the pretrained model and fine-tune it to Sorbian and German. This follows [RE-LM](https://www.aclweb.org/anthology/2020.emnlp-main.214.pdf). Then, we train for NMT in an unsuperivsed way (online backtranslation).

- For the Statistical MT part, we use [monoses](https://github.com/artetxem/monoses). Then, we backtranslate and get a pseudo-parallel corpus for both directions.

- We train for UNMT using online BT and we also have a supervised loss on the pseudo-parallel corpus from USMT

- We sample when doing the prediction during online BT. 

- We oversample the Sorbian corpus using BPE-Dropout. For this, we preprocess data using the advanced features of [subword-nmt](https://github.com/rsennrich/subword-nmt#advanced-features). 

Our proposed pipeline:
<img src="https://github.com/alexandra-chron/umt-lmu-wmt2020/blob/master/system_overview.png" width="380">



#### Reference

```
@article{chronopoulou2020lmu,
      title={The LMU Munich System for the WMT 2020 Unsupervised Machine Translation Shared Task}, 
      author={Alexandra Chronopoulou and Dario Stojanovski and Viktor Hangya and Alexander Fraser},
      url = {https://arxiv.org/abs/2010.13192},
      journal={Proceedings of the Conference on Machine Translation (WMT)},
      year={2020}
}
```


