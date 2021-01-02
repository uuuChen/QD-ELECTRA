# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Pretrain transformer with Masked LM and Sentence Classification """

from random import randint, shuffle
import fire

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from transformers import ElectraForPreTraining, ElectraForMaskedLM, ElectraConfig
from QD_electra_model import DistillELECTRA, QuantizedDistillELECTRA
import QD_electra_model

import tokenization
import optim
import train

from utils import set_seeds, get_device, truncate_tokens_pair

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


def seek_random_offset(f, back_margin=2000):
    """ seek random offset of file pointer """
    f.seek(0, 2)
    # we remain some amount of text to read
    max_offset = f.tell() - back_margin
    f.seek(randint(0, max_offset), 0)
    f.readline() # throw away an incomplete sentence


class SentPairDataLoader():
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, file, batch_size, tokenize, max_len, short_sampling_prob=0.1, pipeline=[]):
        super().__init__()
        self.f_pos = open(file, "r", encoding='utf-8', errors='ignore') # for a positive sample
        self.f_neg = open(file, "r", encoding='utf-8', errors='ignore') # for a negative (random) sample
        self.tokenize = tokenize # tokenize function
        self.max_len = max_len # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.pipeline = pipeline
        self.batch_size = batch_size

    def read_tokens(self, f, length, discard_last_and_restart=True):
        """ Read tokens from file pointer with limited length """
        tokens = []
        while len(tokens) < length:
            line = f.readline()
            if not line: # end of file
                return None
            if not line.strip(): # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = [] # throw all and restart
                    continue
                else:
                    return tokens # return last tokens in the document
            tokens.extend(self.tokenize(line.strip()))
        return tokens

    def __iter__(self): # iterator to load data
        while True:
            batch = []
            instance = None
            for i in range(self.batch_size):
                # sampling length of each tokens_a and tokens_b
                # sometimes sample a short sentence to match between train and test sequences
                # len_tokens = randint(1, int(self.max_len / 2)) \
                #     if rand() < self.short_sampling_prob \
                #     else int(self.max_len / 2)

                tokens_a = self.read_tokens(self.f_pos, self.max_len, True)

                if tokens_a is None: # end of file
                    self.f_pos.seek(0, 0) # reset file pointer
                    return

                for proc in self.pipeline:
                    instance = proc(tokens_a)

                batch.append(instance)

            # To Tensor
            # par_names = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
            # batch_tensors = dict(zip(par_names, [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]))
            batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
            yield batch_tensors


class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Pretrain(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512):
        super().__init__()
        self.max_pred = max_pred # max tokens of prediction
        self.mask_prob = mask_prob # masking probability
        self.vocab_words = vocab_words # vocabulary (sub)words
        self.indexer = indexer # function from token to token index
        self.max_len = max_len

    def __call__(self, tokens_a):
        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, [], self.max_len - 2)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        token_type_ids = [0] * self.max_len
        attention_mask = [1] * len(tokens)

        # Get original input ids as ElectraGenerator labels
        labels = [-100] * self.max_len

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens) * self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = [i for i, token in enumerate(tokens)
                    if token != '[CLS]' and token != '[SEP]']
        shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            attention_mask[pos] = 0
            labels[pos] = self.indexer(tokens[pos])[0]  # get the only element
            tokens[pos] = '[MASK]'

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        attention_mask.extend([0] * n_pad)

        print(input_ids)
        print(attention_mask)
        print(token_type_ids)
        print(labels)

        return (input_ids, attention_mask, token_type_ids, labels)


def main(train_cfg='config/electra_pretrain.json',
         model_cfg='config/electra_base.json',
         data_file='../tbc/books_large_all.txt',
         model_file=None,
         data_parallel=True,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='../exp/electra/pretrain',
         log_dir='../exp/electra/pretrain/runs',
         max_len=128,
         max_pred=20,
         mask_prob=0.15):

    train_cfg = train.Config.from_json(train_cfg)
    model_cfg = QD_electra_model.Config.from_json(model_cfg)

    set_seeds(train_cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [Preprocess4Pretrain(max_pred,
                                    mask_prob,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    max_len)]

    data_iter = SentPairDataLoader(data_file,
                                   train_cfg.batch_size,
                                   tokenize,
                                   max_len,
                                   pipeline=pipeline)

    # Get distilled-electra and quantized-distilled-electra
    generator = ElectraForMaskedLM.from_pretrained('google/electra-small-generator')
    t_discriminator = ElectraForPreTraining.from_pretrained('google/electra-base-discriminator')
    s_discriminator = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
    distillElectra = DistillELECTRA(generator,
                                    t_discriminator,
                                    model_cfg.t_emb_size,
                                    model_cfg.t_dim_ff,
                                    s_discriminator,
                                    model_cfg.s_dim_ff,
                                    model_cfg.s_emb_size)

    optimizer = optim.optim4GPU(train_cfg, distillElectra)
    trainer = train.Trainer(train_cfg, distillElectra, data_iter, optimizer, save_dir, get_device())

    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX

    crossEntropyLoss = nn.CrossEntropyLoss()
    mseLoss = nn.MSELoss()

    def get_distillElectra_loss(model, batch, global_step, train_cfg): # make sure loss is tensor
        input_ids, attention_mask, token_type_ids, labels = batch

        g_outputs, t_d_outputs, s_d_outputs = model(input_ids, attention_mask, token_type_ids, labels)

        # Get original electra loss
        electra_loss = g_outputs.loss + s_d_outputs.loss

        # Get distillation loss
        # 1. teacher and student logits (cross-entropy + temperature)
        # 2. embedding layer loss + hidden losses (MSE)
        # 3. attention matrices loss (MSE)
        soft_logits_loss = crossEntropyLoss(t_d_outputs.logits / train_cfg.temperature,
                                            s_d_outputs.logits / train_cfg.temperature)

        hidden_layers_loss = 0
        for t_hidden, s_hidden in zip(t_d_outputs.hidden_states, s_d_outputs.hidden_states):
            hidden_layers_loss += mseLoss(t_hidden, s_hidden)

        # attention_loss =

        # writer.add_scalars('data/scalar_group',
        #                    {'loss_lm': loss_lm.item(),
        #                     'loss_clsf': loss_clsf.item(),
        #                     'loss_total': (loss_lm + loss_clsf).item(),
        #                     'lr': optimizer.get_lr()[0],
        #                    },
        #                    global_step)
        return electra_loss + soft_logits_loss + hidden_layers_loss

    trainer.train(get_distillElectra_loss, model_file, None, data_parallel)


if __name__ == '__main__':
    fire.Fire(main)


# ------------------------------
# TODO
# 1. 改寫 Trainer.train
# 2. 改寫 get_loss
# 3. 增加 distillation
# ------------------------------
#
# for batch in data_iter:
#
#     generator = ElectraForMaskedLM.from_pretrained('google/electra-small-generator')
#     s_discriminator = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
#     t_discriminator = ElectraForPreTraining.from_pretrained('google/electra-base-discriminator')
#
#     g_outputs = generator(**batch, output_attentions=True, output_hidden_states=True)
#     g_outputs_ids = torch.argmax(g_outputs.logits, axis=2)
#
#     d_labels = (batch['labels'] != g_outputs_ids)
#     s_d_outputs = s_discriminator(g_outputs_ids, labels=d_labels)
#     t_d_outputs = t_discriminator(g_outputs_ids, labels=d_labels)
#
#     print(s_d_outputs.loss, t_d_outputs.loss)
#
#     break