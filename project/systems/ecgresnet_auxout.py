import sys
import os
import torch
from icecream import ic
import pandas as pd
import datetime
from argparse import ArgumentParser
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.core.step_result import TrainResult
from pytorch_lightning.metrics import functional as FM

from uncertainty.network.ecgresnet_auxout import ECGResNet_AuxOut
from ecgnet.utils.loss import SoftmaxFocalLoss
from uncertainty.utils.helpers import create_results_directory
from uncertainty.utils.focalloss_weights import FocalLoss

class ECGResNetAuxOutSystem(pl.LightningModule):

    def __init__(self, in_length, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, n_samples, n_logit_samples, loss_weights=None, 
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.n_samples = n_samples
        self.n_logit_samples = n_logit_samples

        self.IDs = torch.empty(0).type(torch.LongTensor)
        self.predicted_labels = torch.empty(0).type(torch.LongTensor)
        self.correct_predictions = torch.empty(0).type(torch.BoolTensor)
        self.aleatoric_uncertainty = torch.empty(0).type(torch.FloatTensor)

        self.model = ECGResNet_AuxOut(in_length, in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation, n_samples, n_logit_samples)

        if loss_weights is not None:
            weights = torch.tensor(loss_weights, dtype = torch.float)
        else:
            weights = loss_weights

        self.loss = FocalLoss(gamma=1, weights = weights)

    def forward(self, x):
        output1, output2_mean, output2_log_var = self.model(x)
        return output1, output2_mean, output2_log_var

    def training_step(self, batch, batch_idx):
        """Performs a training step.

        Args:
            batch (dict): Output of the dataloader.
            batch_idx (int): Index no. of this batch.

        Returns:
            tensor: Total loss for this step.
        """
        data, target = batch['waveform'], batch['label']

        # Make prediction
        output1, output2_mean, output2_log_var = self(data)

        # Sample from logits, returning a vector x_i
        x_i = self.model.sample_logits(self.n_samples, output2_mean, output2_log_var, average=True)
            
        # Apply softmax to obtain probability vector p_i
        # p_i = F.softmax(x_i, dim=1)

        train_loss1 = self.loss(output1, target)
        train_loss2 = self.loss(x_i, target)
        total_train_loss = (0.3 * train_loss1) + train_loss2
        
        self.log('train_loss', total_train_loss)

        return {'loss': total_train_loss}
    
    def validation_step(self, batch, batch_idx):
        data, target = batch['waveform'], batch['label']

        # Make prediction
        _, output2_mean, output2_log_var = self(data)
            
        # Sample from logits, returning a  vector x_i
        x_i = self.model.sample_logits(self.n_samples, output2_mean, output2_log_var, average=True)
        
        # Apply softmax to obtain probability vector p_i
        p_i = F.softmax(x_i, dim=1)
        
        val_loss = self.loss(x_i, target)
        acc = FM.accuracy(p_i, target)

        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        metrics = {'val_loss': val_loss.item(), 'val_acc': acc.item()}
        self.log('val_acc', acc.item())
        self.log('val_loss', val_loss.item())
        return metrics

    def test_step(self, batch, batch_idx, save_to_csv=False):
        data, target = batch['waveform'], batch['label']

        # Make prediction
        _, output2_mean, output2_log_var = self(data)
            
        # Sample from logits, returning a  vector x_i
        x_i = self.model.sample_logits(self.n_samples, output2_mean, output2_log_var, average=True)
        
        # Apply softmax to obtain probability vector p_i
        p_i = F.softmax(x_i, dim=1)
        
        # Take exponent to get the variance
        output2_var = output2_log_var.exp()

        predicted_labels = p_i.argmax(dim=1)
        correct_predictions = torch.eq(predicted_labels, target)

        # Get the variance of the predicted labels by selecting the variance of
        # the labels with highest average Softmax value
        predicted_labels_var = torch.gather(output2_var, 1, predicted_labels.unsqueeze_(1))[:, 0]
        
        # Get metrics
        test_loss = self.loss(p_i, target)
        acc = FM.accuracy(p_i, target)

        self.log('test_acc', acc.item())
        self.log('test_loss', test_loss.item())

        self.IDs = torch.cat((self.IDs, batch['id']), 0)
        self.predicted_labels = torch.cat((self.predicted_labels, predicted_labels), 0)
        self.aleatoric_uncertainty = torch.cat((self.aleatoric_uncertainty, predicted_labels_var), 0)
        self.correct_predictions = torch.cat((self.correct_predictions, correct_predictions), 0)

        return {'test_loss': test_loss.item(), 'test_acc': acc.item()}

    # Initialize optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='none_auxout')
        parser.add_argument('--n_samples', type=int, default=100)
        parser.add_argument('--n_logit_samples', type=int, default=100)
        parser.add_argument('--ensembling_method', type=bool, default=False)
        return parser

    # Combine results into single dataframe and save to disk
    def save_results(self):
        results = pd.concat([
            pd.DataFrame(self.IDs.numpy(), columns= ['ID']),  
            pd.DataFrame(self.predicted_labels.numpy(), columns= ['predicted_label']),
            pd.DataFrame(self.correct_predictions.numpy(), columns= ['correct_prediction']),
            pd.DataFrame(self.aleatoric_uncertainty.numpy(), columns= ['aleatoric_uncertainty']), 
        ], axis=1)

        create_results_directory()
        results.to_csv('results/{}_{}_results.csv'.format(self.__class__.__name__, datetime.datetime.now().replace(microsecond=0).isoformat()), index=False)
