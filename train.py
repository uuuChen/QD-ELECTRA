# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import os
import json
import numpy as np
from typing import NamedTuple
from tqdm import tqdm
from abc import abstractmethod

import torch
import torch.nn as nn

import checkpoint


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, train_cfg, model_cfg, model, train_data_iter, eval_data_iter, optimizer, save_dir, device):
        self.train_cfg = train_cfg # config for training : see class Config
        self.model_cfg = model_cfg # config for model
        self.model = model
        self.train_data_iter = train_data_iter # iterator to load data
        self.eval_data_iter = eval_data_iter # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device # device name

    def train(self, model_file=None, pretrain_file=None, data_parallel=True):
        """ Train Loop """
        self.model.train() # train mode
        self.load(model_file, pretrain_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.train_cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(self.train_data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                loss = self.get_loss(model, batch, global_step).mean() # mean() for Data
                # Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())

                if global_step % self.train_cfg.save_steps == 0: # save
                    self.save(global_step)

                if self.train_cfg.total_steps and self.train_cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f' % (e+1, self.train_cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step) # save and finish when global_steps reach total_steps
                    return

            print('Epoch %d/%d : Average Loss %5.3f' % (e+1, self.train_cfg.n_epochs, loss_sum/(i+1)))
        self.save(global_step)

    def eval(self, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file, None)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)
        iter_bar = tqdm(self.eval_data_iter, desc='Iter')
        global_step = 0
        result_values_sum = None
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                result_dict = self.evaluate(model, batch) # accuracy to print
                result_values = np.array(list(result_dict.values()))
            global_step += 1
            if result_values_sum is None:
                result_values_sum = [0] * len(result_values)
            result_values_sum += result_values
            iter_bar.set_description('Iter')
            print(list(zip(result_dict.keys(), result_values_sum/global_step)))

    def load(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file, map_location=self.device), strict=False)

        elif pretrain_file: # use pretrained transformer
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                self.model.transformer.load_state_dict({
                    key[12:]: value for key, value in torch.load(pretrain_file).items()
                    if key.startswith('transformer')
                }) # load only transformer parts

    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_steps_' + str(i) + '.pt'))

    @abstractmethod
    def get_loss(self, model, batch, global_step):
        return NotImplementedError

    @abstractmethod
    def evaluate(self, model, batch):
        return NotImplementedError


