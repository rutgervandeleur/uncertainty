import sys
import os
import torch
import pandas as pd
import datetime
from argparse import ArgumentParser
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

from network.ecgresnet_mcdropout import ECGResNet_MCDropout
from utils.helpers import create_results_directory
from utils.focalloss_weights import FocalLoss

class ECGResNetMCDropoutSystem(pl.LightningModule):
    """
    This class implements the ECGResNet with Monte Carlo dropout PyTorch Lightning.
    It can estimate the epistemic uncertainty of its predictions.
    """

    def __init__(self, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, n_dropout_samples, sampling_dropout_rate, loss_weights=None, 
                 **kwargs):
        """
        Initializes the ECGResNetMCDropoutSystem

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
          n_dropout_samples: number of Monte Carlo dropout samples to take
          sampling_dropout_rate: the ratio of dropped-out neurons during Monte Carlo sampling
          loss_weights: array of weights for the loss term
        """

        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.register_buffer('IDs', torch.empty(0).type(torch.LongTensor))
        self.register_buffer('predicted_labels', torch.empty(0).type(torch.LongTensor))
        self.register_buffer('correct_predictions', torch.empty(0).type(torch.BoolTensor))
        self.register_buffer('epistemic_uncertainty', torch.empty(0).type(torch.FloatTensor))

        self.model = ECGResNet_MCDropout(in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation, n_dropout_samples, sampling_dropout_rate)

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
        
        self.log('train_loss', total_train_loss)

        return {'loss': total_train_loss}
    
    def validation_step(self, batch, batch_idx):
        data, target = batch['waveform'], batch['label']
        output1, output2 = self(data)
        val_loss = self.loss(output2.squeeze(), target)
        acc = FM.accuracy(output2.squeeze(), target)

        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        metrics = {'val_loss': val_loss.item(), 'val_acc': acc.item()}
        self.log('val_acc', acc.item())
        self.log('val_loss', val_loss.item())
        return metrics

    def test_step(self, batch, batch_idx, save_to_csv=False):

        # Enable dropout at test time.
        self.model.enable_dropout()
        data, target = batch['waveform'], batch['label']

        # Make prediction with dropout turned on, sampling multiple times.
        samples, sample_mean, sample_var = self.model.mc_sample(data)
        
        # Get predicted labels by choosing the labels with the highest average Softmax value
        predicted_labels = sample_mean.argmax(dim=1)

        # Get the variance of the predicted labels by selecting the variance of
        # the labels with highest average Softmax value
        predicted_labels_var = torch.gather(sample_var, 1, sample_mean.argmax(dim=1).unsqueeze_(1))[:, 0]
        
        # Get metrics
        test_loss = self.loss(sample_mean, target)
        acc = FM.accuracy(sample_mean, target)

        self.log('test_acc', acc.item())
        self.log('test_loss', test_loss.item())

        self.IDs = torch.cat((self.IDs, batch['id']), 0)
        self.predicted_labels = torch.cat((self.predicted_labels, predicted_labels), 0)
        self.epistemic_uncertainty = torch.cat((self.epistemic_uncertainty, predicted_labels_var), 0)
        self.correct_predictions = torch.cat((self.correct_predictions, torch.eq(predicted_labels, target.data)), 0)

        return {'test_loss': test_loss.item(), 'test_acc': acc.item(), 'test_loss': test_loss.item()}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='mcdropout_none')
        parser.add_argument('--n_dropout_samples', type=int, default=20)
        parser.add_argument('--sampling_dropout_rate', type=float, default=0.1)
        parser.add_argument('--ensembling_method', type=bool, default=False)
        return parser

    # Combine results into single dataframe and save to disk
    def save_results(self):
        """
        Combine results into single dataframe and save to disk as .csv file
        """
        results = pd.concat([
            pd.DataFrame(self.IDs.cpu().numpy(), columns= ['ID']),  
            pd.DataFrame(self.predicted_labels.cpu().numpy(), columns= ['predicted_label']),
            pd.DataFrame(self.correct_predictions.cpu().numpy(), columns= ['correct_prediction']),
            pd.DataFrame(self.epistemic_uncertainty.cpu().numpy(), columns= ['epistemic_uncertainty']), 
        ], axis=1)

        create_results_directory()
        results.to_csv('results/{}_{}_results.csv'.format(self.__class__.__name__, datetime.datetime.now().replace(microsecond=0).isoformat()), index=False)
