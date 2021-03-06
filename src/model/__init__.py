# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from .transformer import TransformerModel


logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # reload a pretrained model
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            assert len(s) == 2
            assert all([x == '' or os.path.isfile(x) for x in s])


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))

def modify_params(initial, num_new_dim, param_name):
    if len(initial.size()) == 2:
        # modified = torch.FloatTensor(initial.size(0) + num_new_dim, initial.size(1))

        if param_name == 'embeddings.weight' or param_name == 'lang_embeddings.weight':
            modified = torch.normal(mean=0, std=initial.size(1) ** -0.5, size=(initial.size(0) + num_new_dim, initial.size(1)))
        elif param_name == "pred_layer.proj.weight":
            # modified = torch.normal(0, 1, size=(initial.size(0) + num_new_dim, initial.size(1)))
            modified = torch.FloatTensor(initial.size(0) + num_new_dim, initial.size(1)).uniform_(-1. / initial.size(1), 1. / initial.size(1))
            # modified = torch.normal(0, 1, size=(initial.size(0) + num_new_dim, initial.size(1)))

        modified[:initial.size(0), :] = initial
    else:
        # modified = torch.FloatTensor(initial.size(0) + num_new_dim)
        modified = torch.zeros(initial.size(0) + num_new_dim)
        modified[:initial.size(0)] = initial
    return modified


def build_model(params, dico):
    """
    Build model.
    """
    if params.encoder_only:
        # build
        model = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))['model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            if params.increase_vocab_by > 0:
                logger.info('Fine-tuning model with a different vocabulary. Increasing loaded embeddings size ...')
                for param_name in ['embeddings.weight', 'pred_layer.proj.weight', 'pred_layer.proj.bias']:
                    reloaded[param_name] = modify_params(reloaded[param_name], params.increase_vocab_by, param_name)

            # # HACK to reload models with less layers
            # for i in range(12, 24):
            #     for k in TRANSFORMER_LAYER_PARAMS:
            #         k = k % i
            #         if k in model.state_dict() and k not in reloaded:
            #             logger.warning("Parameter %s not found. Ignoring ..." % k)
            #             reloaded[k] = model.state_dict()[k]

            model.load_state_dict(reloaded, strict=False)

        logger.debug("Model: {}".format(model))
        logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))

        return model.cuda()

    else:
        # build
        encoder = TransformerModel(params, dico, is_encoder=True, with_output=True)  # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
        decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)

        # reload a pretrained model
        if params.reload_model != '':
            enc_path, dec_path = params.reload_model.split(',')
            assert not (enc_path == '' and dec_path == '')

            # reload encoder
            if enc_path != '':
                logger.info("Reloading encoder from %s ..." % enc_path)
                enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
                if all([k.startswith('module.') for k in enc_reload.keys()]):
                    enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}

                if params.increase_vocab_by > 0:
                    logger.info('Fine-tuning model with a different vocabulary. Increasing loaded embeddings size ...')
                    for param_name in ['embeddings.weight', 'pred_layer.proj.weight', 'pred_layer.proj.bias']:
                        enc_reload[param_name] = modify_params(enc_reload[param_name], params.increase_vocab_by, param_name)

                if len(params.mass_steps) > 0:
                    enc_reload['lang_embeddings.weight'] = modify_params(enc_reload['lang_embeddings.weight'], 1, 'lang_embeddings.weight')

                encoder.load_state_dict(enc_reload, strict=False)

            # reload decoder
            if dec_path != '':
                logger.info("Reloading decoder from %s ..." % dec_path)
                dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
                if all([k.startswith('module.') for k in dec_reload.keys()]):
                    dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}

                if params.increase_vocab_by > 0:
                    logger.info('Fine-tuning model with a different vocabulary. Increasing loaded embeddings size ...')
                    for param_name in ['embeddings.weight', 'pred_layer.proj.weight', 'pred_layer.proj.bias']:
                        dec_reload[param_name] = modify_params(dec_reload[param_name], params.increase_vocab_by, param_name)

                if len(params.mass_steps) > 0:
                    dec_reload['lang_embeddings.weight'] = modify_params(dec_reload['lang_embeddings.weight'], 1, 'lang_embeddings.weight')

                decoder.load_state_dict(dec_reload)

        logger.info("Encoder: {}".format(encoder))
        logger.info("Decoder: {}".format(decoder))

        logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        logger.info("Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))

        return encoder.cuda(), decoder.cuda()
