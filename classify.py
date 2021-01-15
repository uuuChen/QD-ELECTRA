# Copyright 2021 Chen-Fa-You, Tseng-Chih-Ying, NTHU
# (Strongly inspired by Intel TinyBert's code and Dong-Hyun Lee's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """

import itertools
import csv
import fire
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tensorboardX import SummaryWriter
from transformers import (
    ElectraConfig,
    BertConfig,
    ElectraForPreTraining,
    ElectraForMaskedLM,
    ElectraForMultipleChoice,
    ElectraForSequenceClassification,
    PretrainedConfig
)

from QDElectra_model import (
    DistillElectraForSequenceClassification,
    QuantizedElectraForSequenceClassification
)
from typing import NamedTuple
import tokenization
import optim
import train

from utils import set_seeds, get_device, truncate_tokens_pair


class CsvDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None

    def __init__(self, file, pipeline=[]): # cvs file and pipeline object
        Dataset.__init__(self)
        data = []
        with open(file, "r", encoding='utf-8') as f:
            # list of splitted lines : line is also list
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for instance in self.get_instances(lines): # instance : tuple of fields
                for proc in pipeline: # a bunch of pre-processing
                    instance = proc(instance)
                data.append(instance)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError


class MRPC(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[0], line[3], line[4] # label, text_a, text_b


class MNLI(CsvDataset):
    """ Dataset class for MNLI """
    labels = ("contradiction", "entailment", "neutral") # label names

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[-1], line[8], line[9] # label, text_a, text_b


class COLA(CsvDataset):
    """Dataset class for COLA"""
    labels = ("0", "1")

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[1], line[3]


class SST2(CsvDataset):
    """Dataset class for SST2"""
    labels = ("0", "1")

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[1], line[0]


class STSB(CsvDataset):
    """Dataset class for STSB"""
    labels = [None]

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[-1], line[7], line[8]


class QQP(CsvDataset):
    """Dataset class for QQP"""
    labels = ("0", "1")

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[5], line[3], line[4]


class QNLI(CsvDataset):
    """Dataset class for QNLI"""
    labels = ("entailment", "not_entailment")

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[-1], line[2], line[1]


class RTE(CsvDataset):
    """Dataset class for RTE"""
    labels = ("entailment", "not_entailment")

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[-1], line[1], line[2]


class WNLI(CsvDataset):
    """Dataset class for WNLI"""
    labels = ("0", "1")

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[-1], line[1], line[2]


def dataset_class(task_name):
    """ Mapping from task_name string to Dataset Class """
    table = {
        'mrpc':  MRPC,
        'mnli':  MNLI,
        'cola':  COLA,
        'sst-2': SST2,
        'sts-b': STSB,
        'qqp':   QQP,
        'qnli':  QNLI,
        'rte':   RTE,
        'wnli':  WNLI
    }
    return table[task_name]


class Pipeline():
    """ Preprocess Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """
    def __init__(self, task_name, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor # e.g. text normalization
        self.tokenize = tokenize # tokenize function
        self.task_name = task_name

    def __call__(self, instance):
        if self.task_name == ("cola" or "sst-2"):
            label, text_a = instance
            text_b = []
        else:
            label, text_a, text_b = instance
        label = self.preprocessor(label)
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) if text_b != [] else []

        return label, tokens_a, tokens_b


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return label, tokens_a, tokens_b


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, output_mode, max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.output_mode = output_mode
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        token_type_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # token type ids
        attention_mask = [1]*(len(tokens_a) + len(tokens_b))
        if self.output_mode == "classification":
            label_id = self.label_map[label]
        elif self.output_mode == "regression":
            label_id = float(label)
        else:
            raise KeyError(output_mode)
        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        token_type_ids.extend([0]*n_pad)
        attention_mask.extend([0]*n_pad)

        return input_ids, attention_mask, token_type_ids, label_id


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def get_task_params(task_name):
    output_modes = {
        "cola":  "classification",
        "mnli":  "classification",
        "mrpc":  "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp":   "classification",
        "qnli":  "classification",
        "rte":   "classification",
        "wnli":  "classification"
    }

    # intermediate distillation default parameters
    default_params = {
        "cola":  {"n_epochs": 50, "max_len": 64},
        "mnli":  {"n_epochs": 5,  "max_len": 128},
        "mrpc":  {"n_epochs": 20, "max_len": 128},
        "sst-2": {"n_epochs": 10, "max_len": 64},
        "sts-b": {"n_epochs": 20, "max_len": 128},
        "qqp":   {"n_epochs": 5,  "max_len": 128},
        "qnli":  {"n_epochs": 10, "max_len": 128},
        "rte":   {"n_epochs": 20, "max_len": 128}
    }
    return output_modes[task_name], default_params[task_name]["n_epochs"], default_params[task_name]["max_len"]


class QuantizedDistillElectraTrainer(train.Trainer):
    def __init__(self, task_name, output_mode, pred_distill, imitate_tinybert, num_labels, writer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.output_mode = output_mode
        self.pred_distill = pred_distill
        self.imitate_tinybert = imitate_tinybert
        self.num_labels = num_labels
        self.writer = writer

        self.bceLoss = nn.BCELoss()
        self.mseLoss = nn.MSELoss()
        self.ceLoss = nn.CrossEntropyLoss()

    def get_loss(self, model, batch, global_step): # make sure loss is tensor
        t_outputs, s_outputs, s2t_hidden_states = model(*batch)

        t_outputs.loss *= self.train_cfg.lambda_
        s_outputs.loss *= self.train_cfg.lambda_
        t_outputs.loss = t_outputs.loss.mean()
        s_outputs.loss = s_outputs.loss.mean()

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
            F.sigmoid(s_outputs.logits / self.train_cfg.temperature),
            F.sigmoid(t_outputs.logits.detach() / self.train_cfg.temperature),
        ) * self.train_cfg.temperature * self.train_cfg.temperature

        hidden_layers_loss = 0
        for t_hidden, s_hidden in zip(t_outputs.hidden_states, s2t_hidden_states):
            hidden_layers_loss += self.mseLoss(s_hidden, t_hidden.detach())

        # -----------------------
        # teacher attention shape per layer : (batch_size, t_n_heads, max_seq_len, max_seq_len)
        # student attention shape per layer : (batch_size, s_n_heads, max_seq_len, max_seq_len)
        # -----------------------
        atten_layers_loss = 0
        split_sections = [self.model_cfg.s_n_heads] * (self.model_cfg.t_n_heads // self.model_cfg.s_n_heads)
        for t_atten, s_atten in zip(t_outputs.attentions, s_outputs.attentions):
            split_t_attens = torch.split(t_atten.detach(), split_sections, dim=1)
            for i, split_t_atten in enumerate(split_t_attens):
                atten_layers_loss += self.mseLoss(s_atten[:, i, :, :], torch.mean(split_t_atten, dim=1))

        if self.imitate_tinybert:
            if not pred_distill:
                total_loss = hidden_layers_loss + atten_layers_loss
            else:
                if self.output_mode == "regression":
                    total_loss = s_outputs.loss
                elif self.output_mode == "classification":
                    total_loss = soft_logits_loss
                else:
                    raise Keyerror(self.output_mode)
        else:
            total_loss = t_outputs.loss + s_outputs.loss + soft_logits_loss + hidden_layers_loss + atten_layers_loss

        self.writer.add_scalars(
            'data/scalar_group', {
                't_discriminator_loss': t_outputs.loss.item(),
                's_discriminator_loss': s_outputs.loss.item(),
                'soft_logits_loss': soft_logits_loss.item(),
                'hidden_layers_loss': hidden_layers_loss.item(),
                'attention_loss': atten_layers_loss.item(),
                'total_loss': total_loss.item(),
                'lr': self.optimizer.get_lr()[0]
            }, global_step
        )

        print(f'\tT-Discriminator Loss {t_outputs.loss.item():.3f}\t'
              f'S-Discriminator Loss {s_outputs.loss.item():.3f}\t'
              f'Soft Logits Loss {soft_logits_loss.item():.3f}\t'
              f'Hidden Loss {hidden_layers_loss.item():.3f}\t'
              f'Attention Loss {atten_layers_loss.item():.3f}\t'
              f'Total Loss {total_loss.item():.3f}\t')

        return total_loss

    def evaluate(self, model, batch):
        _, _, _, label_ids = batch
        _, s_outputs, _ = model(*batch)
        if self.output_mode == "classification":
            loss = self.ceLoss(s_outputs.logits.view(-1, self.num_labels), label_ids).mean().item()
            preds = np.argmax(s_outputs.logits.detach().cpu().numpy(), axis=1)
        elif self.output_mode == "regression":
            loss = self.mseLoss(s_outputs.logits.view(-1), label_ids).mean().item()
            preds = np.squeeze(s_outputs.logits.detach().cpu().numpy())
        else:
            raise KeyError(self.output_mode)
        result = compute_metrics(self.task_name, preds, label_ids.numpy())
        result['loss'] = loss
        return result


def main(task_name='qqp',
         base_train_cfg='config/QDElectra_pretrain.json',
         train_cfg='config/train_mrpc.json',
         model_cfg='config/QDElectra_base.json',
         data_file='GLUE/glue_data/QQP/train.tsv',
         model_file=None,
         data_parallel=True,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         log_dir='../exp/electra/pretrain/runs',
         save_dir='../exp/bert/mrpc',
         pred_distill=True,
         quantize=False,
         imitate_tinybert=False):

    train_cfg_dict = json.load(open(base_train_cfg, "r"))
    train_cfg_dict.update(json.load(open(train_cfg, "r")))
    train_cfg = ElectraConfig().from_dict(train_cfg_dict)
    model_cfg = ElectraConfig().from_json_file(model_cfg)
    output_mode, train_cfg.n_epochs, max_len = get_task_params(task_name)

    set_seeds(train_cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    TaskDataset = dataset_class(task_name) # task dataset class according to the task name
    model_cfg.num_labels = len(TaskDataset.labels)
    pipeline = [
        Tokenizing(task_name, tokenizer.convert_to_unicode, tokenizer.tokenize),
        AddSpecialTokensWithTruncation(max_len),
        TokenIndexing(tokenizer.convert_tokens_to_ids, TaskDataset.labels, output_mode, max_len)
    ]
    data_set = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(data_set, batch_size=train_cfg.batch_size, shuffle=True)

    # generator = ElectraForSequenceClassification.from_pretrained(
    #     'google/electra-small-generator'
    # )
    # t_discriminator = ElectraForSequenceClassification.from_pretrained(
    #     'google/electra-base-discriminator'
    # )
    # s_discriminator = QuantizedElectraForSequenceClassification if quantize else ElectraForSequenceClassification
    # s_discriminator = s_discriminator.from_pretrained(
    #     'google/electra-small-discriminator', config=model_cfg
    # )
    # model = DistillElectraForSequenceClassification(generator, t_discriminator, s_discriminator, model_cfg)
    # model.load_state_dict(
    #     torch.load('./saved_QDElectra/model_steps_40000.pt', map_location=get_device()), strict=False
    # )
    # model.t_discriminator.save_pretrained("load_model/teacher/")
    # model.s_discriminator.save_pretrained("load_model/student/")
    # raise Exception
    teacher_config = ElectraConfig().from_json_file("load_model/teacher/config.json")
    student_config = ElectraConfig().from_json_file("load_model/student/config.json")
    generator = ElectraForSequenceClassification.from_pretrained(
        'google/electra-small-generator'
    )
    t_discriminator = ElectraForSequenceClassification.from_pretrained(
        'load_model/teacher/pytorch_model.bin', config=teacher_config
    )
    s_discriminator = QuantizedElectraForSequenceClassification if quantize else ElectraForSequenceClassification
    s_discriminator = s_discriminator.from_pretrained(
        'load_model/student/pytorch_model.bin', config=student_config
    )

    model = DistillElectraForSequenceClassification(generator, t_discriminator, s_discriminator, model_cfg)

    optimizer = optim.optim4GPU(train_cfg, model)
    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX

    base_trainer_args = (train_cfg, model_cfg, model, data_iter, optimizer, save_dir, get_device())
    trainer = QuantizedDistillElectraTrainer(
        task_name, output_mode, pred_distill, imitate_tinybert, len(TaskDataset.labels), writer, *base_trainer_args
    )

    trainer.train(model_file, None, data_parallel)
    trainer.eval(model_file, data_parallel)


if __name__ == '__main__':
    fire.Fire(main)
