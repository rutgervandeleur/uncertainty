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

from network.ecgresnet_auxout import ECGResNet_AuxOut
from utils.helpers import create_results_directory
from utils.focalloss_weights import FocalLoss

class ECGResNetEnsemble_AuxOutSystem(pl.LightningModule):
    """
    This class implements the ECGResNet with ensemble and auxiliary output in PyTorch Lightning.
    It can estimate the epistemic and aleatoric uncertainty of its predictions.
    """

    def __init__(self, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, ensemble_size, n_logit_samples, loss_weights=None, 
                 **kwargs):
        """
        Initializes the ECGResNetEnsemble_AuxOutSystem

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
          n_logit_samples: number of logit samples of the auxiliary output
          loss_weights: array of weights for the loss term
        """

        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.ensemble_size = ensemble_size
        self.n_logit_samples = n_logit_samples

        self.register_buffer('IDs', torch.empty(0).type(torch.LongTensor))
        self.register_buffer('predicted_labels', torch.empty(0).type(torch.LongTensor))
        self.register_buffer('correct_predictions', torch.empty(0).type(torch.BoolTensor))
        self.register_buffer('aleatoric_uncertainty', torch.empty(0).type(torch.FloatTensor))
        self.register_buffer('epistemic_uncertainty', torch.empty(0).type(torch.FloatTensor))
        self.register_buffer('total_uncertainty', torch.empty(0).type(torch.FloatTensor))

        self.models = []
        self.optimizers = []

        # Device needs to be selected manually because PyTorch Lightning does not
        # recognize multiple models when in list
        manual_device = torch.device('cuda' if torch.cuda.is_available() and kwargs['gpus'] != 0 else 'cpu')

        for i in range(self.ensemble_size):
            self.models.append(ECGResNet_AuxOut(in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation).to(manual_device))

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
            output1: Output at the auxiliary point of the ensemble member
            output2: Output at the end of the ensemble member
            output2_log_var: The log variance of the ensemble_member
        """

        output1, output2_mean, output2_log_var = self.models[model_idx](x)
            
        return output1, output2_mean, output2_log_var

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
        for model_idx in range(self.ensemble_size):
            # Make prediction
            output1, output2_mean, output2_log_var = self(data, model_idx)

            # Sample from logits, returning a vector x_i
            x_i = self.models[model_idx].sample_logits(self.n_logit_samples, output2_mean, output2_log_var, average=True)

            train_loss1 = self.loss(output1, target)
            train_loss2 = self.loss(x_i, target)
            total_train_loss = (0.3 * train_loss1) + train_loss2

            # Update weights for each model using individual optimizers
            self.manual_backward(total_train_loss, self.optimizers[model_idx])
            self.optimizers[model_idx].step()
            self.optimizers[model_idx].zero_grad()
            losses.append(total_train_loss.item())

            self.log('model_{}_train_loss'.format(model_idx), total_train_loss)

        average_train_loss = np.mean(losses)
        self.log('average_train_loss', average_train_loss)

        return {'loss': average_train_loss}

    def validation_step(self, batch, batch_idx):
        prediction_individual = torch.empty(batch['label'].shape[0], self.ensemble_size, self.num_classes)

        data, target = batch['waveform'], batch['label']

        # Predict for each model
        for model_idx in range(self.ensemble_size):
            # Make prediction
            _, output2_mean, output2_log_var = self(data, model_idx)

            # Sample from logits, returning avector x_i
            x_i = self.models[model_idx].sample_logits(self.n_logit_samples, output2_mean, output2_log_var, average=True)

            prediction_individual[:, model_idx] = x_i
            
        # Calculate mean over predictions from individual ensemble members
        prediction_ensemble_mean = F.softmax(torch.mean(prediction_individual, dim=1), dim=1).type_as(data)
    
        val_loss = self.loss(prediction_ensemble_mean, target)
        acc = FM.accuracy(prediction_ensemble_mean, target)

        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        metrics = {'val_loss': val_loss.item(), 'val_acc': acc.item()}
        self.log('val_acc', acc.item())
        self.log('val_loss', val_loss.item())
        return metrics

    def test_step(self, batch, batch_idx, save_to_csv=False):

        prediction_individual = torch.empty(batch['label'].shape[0], self.ensemble_size, self.num_classes)
        aleatoric_var = torch.empty(batch['label'].shape[0], self.ensemble_size, self.num_classes)
        data, target = batch['waveform'], batch['label']

        # Predict for each model
        for model_idx, model in enumerate(self.models):

            # Make prediction
            _, output2_mean, output2_log_var = self(data, model_idx)

            # Sample from logits, returning a  vector x_i
            x_i = self.models[model_idx].sample_logits(self.n_logit_samples, output2_mean, output2_log_var, average=True)

            prediction_individual[:, model_idx] = x_i.data

            # Take exponent to get the variance
            output2_var = output2_log_var.exp()
            aleatoric_var[:, model_idx] = output2_var.data
            
        # Calculate mean and variance over predictions from individual ensemble members
        prediction_ensemble_mean = F.softmax(torch.mean(prediction_individual, dim=1), dim=1).type_as(data)
        prediction_ensemble_var = torch.var(prediction_individual, dim=1).type_as(data)

        # Get the average aleatoric uncertainty for each prediction
        prediction_aleatoric_var = torch.mean(aleatoric_var, dim=1).type_as(data)

        # Select the predicted labels
        predicted_labels = prediction_ensemble_mean.argmax(dim=1)

        test_loss = self.loss(prediction_ensemble_mean, target)
        acc = FM.accuracy(prediction_ensemble_mean, target)

        # Get the epistemic variance of the predicted labels by selecting the variance of
        # the labels with highest average Softmax value
        predicted_labels_var = torch.gather(prediction_ensemble_var, 1, prediction_ensemble_mean.argmax(dim=1).unsqueeze_(1))[:, 0]

        # Get the aleatoric variance of the predicted labels by selecting the variance of
        # the labels with highest average Softmax value
        predicted_labels_aleatoric_var = torch.gather(prediction_aleatoric_var, 1, prediction_ensemble_mean.argmax(dim=1).unsqueeze_(1))[:, 0]

        total_var = predicted_labels_var + predicted_labels_aleatoric_var
        
        # Log and save metrics
        self.log('test_acc', acc.item())
        self.log('test_loss', test_loss.item())

        self.IDs = torch.cat((self.IDs, batch['id']), 0)
        self.predicted_labels = torch.cat((self.predicted_labels, predicted_labels), 0)
        self.epistemic_uncertainty = torch.cat((self.epistemic_uncertainty, predicted_labels_var), 0)
        self.aleatoric_uncertainty = torch.cat((self.aleatoric_uncertainty, predicted_labels_aleatoric_var), 0)
        self.total_uncertainty = torch.cat((self.total_uncertainty, total_var), 0)
        self.correct_predictions = torch.cat((self.correct_predictions, torch.eq(predicted_labels, target.data)), 0)

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
        parser.add_argument('--n_logit_samples', type=int, default=100)
        return parser

    def save_results(self):
        """
        Combine results into single dataframe and save to disk as .csv file
        """
        results = pd.concat([
            pd.DataFrame(self.IDs.cpu().numpy(), columns= ['ID']),  
            pd.DataFrame(self.predicted_labels.cpu().numpy(), columns= ['predicted_label']),
            pd.DataFrame(self.correct_predictions.cpu().numpy(), columns= ['correct_prediction']),
            pd.DataFrame(self.epistemic_uncertainty.cpu().numpy(), columns= ['epistemic_uncertainty']), 
            pd.DataFrame(self.aleatoric_uncertainty.cpu().numpy(), columns= ['aleatoric_uncertainty']), 
            pd.DataFrame(self.total_uncertainty.cpu().numpy(), columns= ['total_uncertainty']), 
        ], axis=1)

        create_results_directory()
        results.to_csv('results/{}_{}_results.csv'.format(self.__class__.__name__, datetime.datetime.now().replace(microsecond=0).isoformat()), index=False)
