export CUDA_VISIBLE_DEVICES=2; export NGPU=1; python3 -m torch.distributed.launch --nproc_per_node=$NGPU train.py --exp_name de_mass_ft_hsb --dump_path /mounts/data/proj/dario/data/unsup-wmt20/dumped/mass/ --data_path /mounts/data/proj/achron/data/de-sorbian-wmt/ --lgs 'de-hsb' --mass_steps 'de,hsb' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size 50000 --max_epoch 100000 --word_mass 0.5 --min_len 5 --reload_model '/mounts/data/proj/dario/data/unsup-wmt20/dumped/mass/de_mass/3w8dqrykpd/checkpoint.pth,/mounts/data/proj/dario/data/unsup-wmt20/dumped/mass/de_mass/3w8dqrykpd/checkpoint.pth' --increase_vocab_for_lang de --increase_vocab_from_lang hsb --increase_vocab_by 15634  # --amp 1
