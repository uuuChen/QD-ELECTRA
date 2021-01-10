# Copyright 2021 Chen-Fa-You, Tseng-Chih-Ying, NTHU
# (Strongly inspired by original Google BERT code, Hugging Face's code and Dong-Hyun Lee's code)

""" Pretrain QD-ELECTRA with Masked LM """

from random import randint, shuffle
import fire
import json
from typing import NamedTuple
from tqdm import tqdm

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from transformers import ElectraForPreTraining, ElectraForMaskedLM
from QDElectra_model import DistillElectraForPreTraining, QuantizedElectraForPreTraining
import torch.nn.functional as F
from transformers import ElectraConfig

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


class QuantizedDistillElectraTrainConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    temperature: int = 1 # temperature for QD-electra logit loss
    lambda_: int = 50 # lambda for QD-electra discriminator loss

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class SentPairDataLoader():
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, file, batch_size, tokenize, max_len, short_sampling_prob=0.1, pipeline=[]):
        super().__init__()
        self.file = open(file, "r", encoding='utf-8', errors='ignore')
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

                tokens_a = self.read_tokens(self.file, self.max_len, True)

                if tokens_a is None: # end of file
                    self.file.seek(0, 0) # reset file pointer
                    return

                for proc in self.pipeline:
                    instance = proc(tokens_a)

                batch.append(instance)

            # To Tensor
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
        # -2  for special tokens [CLS], [SEP]
        truncate_tokens_pair(tokens_a, [], self.max_len - 2)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        token_type_ids = [0] * self.max_len
        attention_mask = [1] * len(tokens)
        original_attention_mask = attention_mask.copy()

        # Get ElectraGenerator label. "-100" means the corresponding token is unmasked, else means the masked token ids
        g_label = [-100] * self.max_len

        # Get original input ids as ElectraDiscriminator labels
        original_input_ids = self.indexer(tokens)

        # For masked Language Models
        # The number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens) * self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = [i for i, token in enumerate(tokens)
                    if token != '[CLS]' and token != '[SEP]']
        shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            attention_mask[pos] = 0
            g_label[pos] = self.indexer(tokens[pos])[0]  # get the only one element from list
            tokens[pos] = '[MASK]'

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        attention_mask.extend([0] * n_pad)

        return input_ids, attention_mask, token_type_ids, g_label, original_input_ids, original_attention_mask


class QuantizedDistillElectraTrainer(train.Trainer):
    def __init__(self, writer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer
        self.bceLoss = nn.BCELoss()
        self.mseLoss = nn.MSELoss()

    def get_loss(self, model, batch, global_step, train_cfg, model_cfg):
        g_outputs, t_d_outputs, s_d_outputs, s2t_hidden_states = model(*batch)

        # Get original electra loss
        t_d_outputs.loss *= train_cfg.lambda_
        s_d_outputs.loss *= train_cfg.lambda_
        electra_loss = g_outputs.loss + s_d_outputs.loss + t_d_outputs.loss

        # -----------------------
        # Get distillation loss
        # 1. teacher and student logits (cross-entropy + temperature)
        # 2. embedding layer loss + hidden losses (MSE)
        # 3. attention matrices loss (MSE)
        #       We only consider :
        #       3-1. Teacher layer numbers equal to student layer numbers
        #       3-2. Teacher head numbers are divisible by Student head numbers
        # -----------------------
        soft_logits_loss = self.bceLoss(
            F.sigmoid(s_d_outputs.logits / train_cfg.temperature),
            F.sigmoid(t_d_outputs.logits.detach() / train_cfg.temperature),
        ) * train_cfg.temperature * train_cfg.temperature

        hidden_layers_loss = 0
        for t_hidden, s_hidden in zip(t_d_outputs.hidden_states, s2t_hidden_states):
            hidden_layers_loss += self.mseLoss(s_hidden, t_hidden.detach())

        # -----------------------
        # teacher attention shape per layer : (batch_size, t_n_heads, max_seq_len, max_seq_len)
        # student attention shape per layer : (batch_size, s_n_heads, max_seq_len, max_seq_len)
        # -----------------------
        atten_layers_loss = 0
        split_sections = [model_cfg.s_n_heads] * (model_cfg.t_n_heads // model_cfg.s_n_heads)
        for t_atten, s_atten in zip(t_d_outputs.attentions, s_d_outputs.attentions):
            split_t_attens = torch.split(t_atten.detach(), split_sections, dim=1)
            for i, split_t_atten in enumerate(split_t_attens):
                atten_layers_loss += self.mseLoss(s_atten[:, i, :, :], torch.mean(split_t_atten, dim=1))

        total_loss = electra_loss + soft_logits_loss + hidden_layers_loss + atten_layers_loss

        self.writer.add_scalars(
            'data/scalar_group', {
                'generator_loss': g_outputs.loss.item(),
                't_discriminator_loss': t_d_outputs.loss.item(),
                's_discriminator_loss': s_d_outputs.loss.item(),
                'soft_logits_loss': soft_logits_loss.item(),
                'hidden_layers_loss': hidden_layers_loss.item(),
                'attention_loss': atten_layers_loss.item(),
                'total_loss': total_loss.item(),
                'lr': self.optimizer.get_lr()[0]
            }, global_step
        )

        print(f'\tGenerator Loss {g_outputs.loss.item():.3f}\t'
              f'T-Discriminator Loss {t_d_outputs.loss.item():.3f}\t'
              f'S-Discriminator Loss {s_d_outputs.loss.item():.3f}\t'
              f'Soft Logits Loss {soft_logits_loss.item():.3f}\t'
              f'Hidden Loss {hidden_layers_loss.item():.3f}\t'
              f'Attention Loss {atten_layers_loss.item():.3f}\t'
              f'Total Loss {total_loss.item():.3f}\t')

        return total_loss

    def _eval(self, data_parallel=True):
        """ Only used to test the correctness of inference-phase of QuantizedElectra """
        self.model.eval()  # evaluation mode
        model = self.model.to(self.device)
        if data_parallel:  # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = []  # prediction results
        iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
        global_step = 0
        for batch in iter_bar:
            global_step += 1
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():  # evaluation without gradient calculation
                loss = self.get_loss(model, batch, global_step, self.train_cfg, self.model_cfg).mean()  # mean() for Data
            results.append(loss)

            iter_bar.set_description('Iter(loss=%5.3f)' % loss)
        return results


def main(train_cfg='config/electra_pretrain.json',
         model_cfg='config/electra_small.json',
         data_file='../tbc/books_large_all.txt',
         model_file=None,
         data_parallel=True,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='../exp/electra/pretrain',
         log_dir='../exp/electra/pretrain/runs',
         max_len=128,
         max_pred=20,
         mask_prob=0.15,
         quantize=False):

    train_cfg = QuantizedDistillElectraTrainConfig.from_json(train_cfg)
    model_cfg = ElectraConfig().from_json_file(model_cfg)

    set_seeds(train_cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [
        Preprocess4Pretrain(
            max_pred, mask_prob, list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, max_len
        )
    ]

    data_iter = SentPairDataLoader(data_file, train_cfg.batch_size, tokenize, max_len, pipeline=pipeline)

    # Get distilled-electra and quantized-distilled-electra
    generator = ElectraForMaskedLM.from_pretrained('google/electra-small-generator')
    t_discriminator = ElectraForPreTraining.from_pretrained('google/electra-base-discriminator')
    s_discriminator = QuantizedElectraForPreTraining(model_cfg) if quantize else ElectraForPreTraining
    s_discriminator = s_discriminator.from_pretrained('google/electra-small-discriminator', config=model_cfg)
    model = DistillElectraForPreTraining(generator, t_discriminator, s_discriminator, model_cfg)

    optimizer = optim.optim4GPU(train_cfg, model)
    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX

    base_trainer_args = (train_cfg, model_cfg, model, data_iter, optimizer, save_dir, get_device())
    trainer = QuantizedDistillElectraTrainer(writer, *base_trainer_args)
    trainer.train(model_file, None, data_parallel)
    trainer._eval()


if __name__ == '__main__':
    fire.Fire(main)
