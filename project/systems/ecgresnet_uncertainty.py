import sys
import os
import torch
from argparse import ArgumentParser
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

from network.ecgresnet import ECGResNet
from utils.focalloss_weights import FocalLoss

class ECGResNetUncertaintySystem(pl.LightningModule):

    def __init__(self, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, loss_weights=None, 
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.model = ECGResNet(in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation)
        if loss_weights is not None:
            weights = torch.tensor(loss_weights, dtype = torch.float)
        else:
            weights = loss_weights

        self.loss = FocalLoss(gamma=1, weights = weights)

    def forward(self, x):
            output1, output2 = self.model(x)
            return output1, output2

    def training_step(self, batch, batch_idx):
        """Performs a training step.

        Args:
            batch (dict): Output of the dataloader.
            batch_idx (int): Index no. of this batch.

        Returns:
            tensor: Total loss for this step.
        """
        data, target = batch['waveform'], batch['label']
        output1, output2 = self(data)
        train_loss1 = self.loss(output1.squeeze(), target)
        train_loss2 = self.loss(output2.squeeze(), target)
        total_train_loss = (0.3 * train_loss1) + train_loss2
        self.log('train_loss', total_train_loss, on_epoch=False)
        return total_train_loss

    def validation_step(self, batch, batch_idx):
        data, target = batch['waveform'], batch['label']
        output1, output2 = self(data)
        val_loss = self.loss(output2.squeeze(), target)
        acc = FM.accuracy(output2.squeeze(), target)

        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        metrics = {'val_acc': acc.item(), 'val_loss': val_loss.item()}
        self.log_dict(metrics)
        # self.log(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        print(metrics)
        # self.log(metrics)
        self.log_dict(metrics)

    def configure_optimizers(self):

        # Initialize optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='none_none')
        parser.add_argument('--ensembling_method', type=bool, default=False)
        return parser

    def save_results(self):
        pass
        
