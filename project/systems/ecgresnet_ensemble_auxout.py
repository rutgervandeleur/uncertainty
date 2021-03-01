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
from icecream import ic

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

from network.ecgresnet_auxout import ECGResNet_AuxOut
from utils.helpers import create_results_directory
from utils.focalloss_weights import FocalLoss

class ECGResNetEnsemble_AuxOutSystem(pl.LightningModule):

    def __init__(self, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, ensemble_size, n_samples, n_logit_samples, loss_weights=None, 
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.ensemble_size = ensemble_size
        self.n_samples = n_samples
        self.n_logit_samples = n_logit_samples

        self.IDs = torch.empty(0).type(torch.LongTensor)
        self.predicted_labels = torch.empty(0).type(torch.LongTensor)
        self.correct_predictions = torch.empty(0).type(torch.BoolTensor)
        self.epistemic_uncertainty = torch.empty(0).type(torch.FloatTensor)
        self.aleatoric_uncertainty = torch.empty(0).type(torch.FloatTensor)
        self.total_uncertainty = torch.empty(0).type(torch.FloatTensor)

        self.models = []
        self.optimizers = []
        for i in range(self.ensemble_size):
            self.models.append(ECGResNet_AuxOut(in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation)
                              )

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
            x_i = self.models[model_idx].sample_logits(self.n_samples, output2_mean, output2_log_var, average=True)

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
            x_i = self.models[model_idx].sample_logits(self.n_samples, output2_mean, output2_log_var, average=True)

            prediction_individual[:, model_idx] = x_i
            
        # Calculate mean over predictions from individual ensemble members
        prediction_ensemble_mean = F.softmax(torch.mean(prediction_individual, dim=1), dim=1)
    
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
            x_i = self.models[model_idx].sample_logits(self.n_samples, output2_mean, output2_log_var, average=True)

            prediction_individual[:, model_idx] = x_i.data

            # Take exponent to get the variance
            output2_var = output2_log_var.exp()
            aleatoric_var[:, model_idx] = output2_var.data
            
        # Calculate mean and variance over predictions from individual ensemble members
        prediction_ensemble_mean = F.softmax(torch.mean(prediction_individual, dim=1), dim=1)
        prediction_ensemble_var = torch.var(prediction_individual, dim=1)

        # Get the average aleatoric uncertainty for each prediction
        prediction_aleatoric_var = torch.mean(aleatoric_var, dim=1)

        # Select the predicted labels
        predicted_labels = prediction_ensemble_mean.argmax(dim=1)

        test_loss = self.loss(prediction_ensemble_mean, target)
        acc = FM.accuracy(prediction_ensemble_mean, target)

        # Get the epistemic variance of the predicted labels by selecting the variance of
        # the labels with highest average Softmax value
        predicted_labels_var = torch.gather(prediction_ensemble_var, 1, prediction_ensemble_mean.argmax(dim=1).unsqueeze_(1))[:, 0].cpu()

        # Get the aleatoric variance of the predicted labels by selecting the variance of
        # the labels with highest average Softmax value
        predicted_labels_aleatoric_var = torch.gather(prediction_aleatoric_var, 1, prediction_ensemble_mean.argmax(dim=1).unsqueeze_(1))[:, 0].cpu()

        total_var = predicted_labels_var + predicted_labels_aleatoric_var
        
        # Log and save metrics
        self.log('test_acc', acc.item())
        self.log('test_loss', test_loss.item())

        self.IDs = torch.cat((self.IDs, batch['id']), 0)
        self.predicted_labels = torch.cat((self.predicted_labels, predicted_labels), 0)
        self.epistemic_uncertainty = torch.cat((self.epistemic_uncertainty, predicted_labels_var), 0)
        self.aleatoric_uncertainty = torch.cat((self.aleatoric_uncertainty, predicted_labels_aleatoric_var), 0)
        self.total_uncertainty = torch.cat((self.total_uncertainty, total_var), 0)
        self.correct_predictions = torch.cat((self.correct_predictions, torch.eq(predicted_labels, target.data.cpu())), 0)

        return {'test_loss': test_loss.item(), 'test_acc': acc.item(), 'test_loss': test_loss.item()}

    # Initialize an optimizer for each model in the ensemble
    def configure_optimizers(self):
        for i in range(self.ensemble_size):
            self.optimizers.append(optim.Adam(self.models[i].parameters(), lr=self.learning_rate))
        
        return self.optimizers

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='ensemble_none')
        parser.add_argument('--ensemble_size', type=int, default=5)
        parser.add_argument('--ensembling_method', type=bool, default=True)
        parser.add_argument('--n_samples', type=int, default=100)
        parser.add_argument('--n_logit_samples', type=int, default=100)
        return parser

    # Combine results into single dataframe and save to disk
    def save_results(self):
        results = pd.concat([
            pd.DataFrame(self.IDs.numpy(), columns= ['ID']),  
            pd.DataFrame(self.predicted_labels.numpy(), columns= ['predicted_label']),
            pd.DataFrame(self.correct_predictions.numpy(), columns= ['correct_prediction']),
            pd.DataFrame(self.epistemic_uncertainty.numpy(), columns= ['epistemic_uncertainty']), 
            pd.DataFrame(self.aleatoric_uncertainty.numpy(), columns= ['aleatoric_uncertainty']), 
            pd.DataFrame(self.total_uncertainty.numpy(), columns= ['total_uncertainty']), 
        ], axis=1)

        create_results_directory()
        results.to_csv('results/{}_{}_results.csv'.format(self.__class__.__name__, datetime.datetime.now().replace(microsecond=0).isoformat()), index=False)
