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
from utils_.focalloss_weights import FocalLoss

class ECGResNetUncertaintySystem(pl.LightningModule):
    """
    This class implements the ECGResNet in PyTorch Lightning.
    """
    def __init__(self, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, loss_weights=None, include_classification = True,
                 **kwargs):
        """
        Initializes the ECGResNetUncertaintySystem

        Args:
          in_channels: number of channels of input
          n_grps: number of ResNet groups
          N: number of blocks per groups
          num_classes: number of classes of the classification problem
          dropout: probability of an argument to get zeroed in the dropout layer
          first_width: width of the first input
          stride: tuple with stride value per block per group
          dilation: spacing between the kernel points of the convolutional layers
          learning_rate: the learning rate of the model
          loss_weights: array of weights for the loss term
        """
 
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.model = ECGResNet(in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation, include_classification)
        if loss_weights is not None:
            weights = torch.tensor(loss_weights, dtype = torch.float)
        else:
            weights = loss_weights

        self.loss = FocalLoss(gamma=1, weights = weights)

    def forward(self, x):
        """
        Performs a forward through the model.

        Args:
            x (tensor): Input data.

        Returns:
            output1: output at the auxiliary point of the ECGResNet
            output2: output at the end of the model
        """
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
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='none_none')
        parser.add_argument('--ensembling_method', type=bool, default=False)
        return parser

    def save_results(self):
        pass
