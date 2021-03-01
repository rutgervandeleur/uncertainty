import sys
import os
import torch
import pandas as pd
import datetime
from argparse import ArgumentParser
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

from network.ecgresnet import ECGResNet
from utils.helpers import create_results_directory
from utils.focalloss_weights import FocalLoss

class ECGResNetEnsembleSystem(pl.LightningModule):
    """
    This class implements an ensemble of ECGResNets in PyTorch Lightning.
    It can estimate the epistemic uncertainty of its predictions.
    """

    def __init__(self, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, ensemble_size, loss_weights=None, 
                 **kwargs):
        """
        Initializes the ECGResNetEnsembleSystem

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
          ensemble_size: the number of models that make up the ensemble
          loss_weights: array of weights for the loss term
        """


        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.ensemble_size = ensemble_size

        self.IDs = torch.empty(0).type(torch.LongTensor)
        self.predicted_labels = torch.empty(0).type(torch.LongTensor)
        self.correct_predictions = torch.empty(0).type(torch.BoolTensor)
        self.epistemic_uncertainty = torch.empty(0).type(torch.FloatTensor)

        self.models = []
        self.optimizers = []

        # Initialize mutliple ensemble members
        for i in range(self.ensemble_size):
            self.models.append(ECGResNet(in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation))

        if loss_weights is not None:
            weights = torch.tensor(loss_weights, dtype = torch.float)
        else:
            weights = loss_weights

        self.loss = FocalLoss(gamma=1, weights = weights)

    def forward(self, x, model_idx):
        """Performs a forward through a single ensemble member.

        Args:
            x (tensor): Input data.
            model_idx (int): Index of the ensemble member.

        Returns:
            Output1: Output at the auxiliary point of the ensemble member
            Output2: Output at the end of the ensemble member
        """
        output1, output2 = self.models[model_idx](x)
            
        return output1, output2

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Performs a training step for all ensemble members.

        Args:
            batch (dict): Output of the dataloader.
            batch_idx (int): Index no. of this batch.

        Returns:
            tensor: Total loss for this step.
        """
        data, target = batch['waveform'], batch['label']

        losses = []
        for i in range(self.ensemble_size):
            output1, output2 = self(data, i)
            train_loss1 = self.loss(output1.squeeze(), target)
            train_loss2 = self.loss(output2.squeeze(), target)

            # Calculate the loss for each model
            total_train_loss = (0.3 * train_loss1) + train_loss2

            # Update weights for each model using individual optimizers
            self.manual_backward(total_train_loss, self.optimizers[i])
            self.optimizers[i].step()
            self.optimizers[i].zero_grad()
            losses.append(total_train_loss.item())

            self.log('model_{}_train_loss'.format(i), total_train_loss)

        average_train_loss = np.mean(losses)
        self.log('average_train_loss', average_train_loss)

        return {'loss': average_train_loss}

    def validation_step(self, batch, batch_idx):
        prediction_individual = torch.empty(batch['label'].shape[0], self.ensemble_size, self.num_classes)

        data, target = batch['waveform'], batch['label']

        # Predict for each model
        for i, model in enumerate(self.models):
            output1, output2 = self(data, i)

            prediction_individual[:, i] = output2.data
            
        # Calculate mean and variance over predictions from individual ensemble members
        prediction_ensemble_mean = F.softmax(torch.mean(prediction_individual, dim=1), dim=1)
        prediction_ensemble_var = torch.var(prediction_individual, dim=1)
    
        val_loss = self.loss(prediction_ensemble_mean, target)
        acc = FM.accuracy(prediction_ensemble_mean, target)

        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        metrics = {'val_loss': val_loss.item(), 'val_acc': acc.item()}
        self.log('val_acc', acc.item())
        self.log('val_loss', val_loss.item())
        return metrics

    def test_step(self, batch, batch_idx, save_to_csv=False):
        prediction_individual = torch.empty(batch['label'].shape[0], self.ensemble_size, self.num_classes)
        data, target = batch['waveform'], batch['label']

        # Predict for each model
        for i, model in enumerate(self.models):

            output1, output2 = self(data, i)
            prediction_individual[:, i] = output2.data
            
        # Calculate mean and variance over predictions from individual ensemble members
        prediction_ensemble_mean = F.softmax(torch.mean(prediction_individual, dim=1), dim=1)
        prediction_ensemble_var = torch.var(prediction_individual, dim=1)
    
        test_loss = self.loss(prediction_ensemble_mean, target)
        acc = FM.accuracy(prediction_ensemble_mean, target)

        # Get the variance of the predicted labels by selecting the variance of
        # the labels with highest average Softmax value
        predicted_labels_var = torch.gather(prediction_ensemble_var, 1, prediction_ensemble_mean.argmax(dim=1).unsqueeze_(1))[:, 0].cpu()
        predicted_labels = prediction_ensemble_mean.argmax(dim=1)
        
        # Log and save metrics
        self.log('test_acc', acc.item())
        self.log('test_loss', test_loss.item())

        self.IDs = torch.cat((self.IDs, batch['id']), 0)
        self.predicted_labels = torch.cat((self.predicted_labels, predicted_labels), 0)
        self.epistemic_uncertainty = torch.cat((self.epistemic_uncertainty, predicted_labels_var), 0)
        self.correct_predictions = torch.cat((self.correct_predictions, torch.eq(predicted_labels, target.data.cpu())), 0)

        return {'test_loss': test_loss.item(), 'test_acc': acc.item(), 'test_loss': test_loss.item()}

    def configure_optimizers(self):
        """
        Initialize an optimizer for each model in the ensemble
        """
        for i in range(self.ensemble_size):
            self.optimizers.append(optim.Adam(self.models[i].parameters(), lr=self.learning_rate))
        
        return self.optimizers

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='ensemble_none')
        parser.add_argument('--ensemble_size', type=int, default=5)
        parser.add_argument('--ensembling_method', type=bool, default=True)
        return parser

    def save_results(self):
        """
        Combine results into single dataframe and save to disk as .csv file
        """
        results = pd.concat([
            pd.DataFrame(self.IDs.numpy(), columns= ['ID']),  
            pd.DataFrame(self.predicted_labels.numpy(), columns= ['predicted_label']),
            pd.DataFrame(self.correct_predictions.numpy(), columns= ['correct_prediction']),
            pd.DataFrame(self.epistemic_uncertainty.numpy(), columns= ['epistemic_uncertainty']), 
        ], axis=1)

        create_results_directory()
        results.to_csv('results/{}_{}_results.csv'.format(self.__class__.__name__, datetime.datetime.now().replace(microsecond=0).isoformat()), index=False)
