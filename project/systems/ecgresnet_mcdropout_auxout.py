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

from network.ecgresnet_mcdropout_auxout import ECGResNet_MCDropout_AuxOutput
from utils.helpers import create_results_directory
from utils.focalloss_weights import FocalLoss

class ECGResNetMCDropout_AuxOutSystem(pl.LightningModule):

    def __init__(self, in_length, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, n_dropout_samples, n_logit_samples, sampling_dropout_rate, loss_weights=None, 
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.n_dropout_samples = n_dropout_samples
        self.n_logit_samples = n_logit_samples

        self.IDs = torch.empty(0).type(torch.LongTensor)
        self.predicted_labels = torch.empty(0).type(torch.LongTensor)
        self.correct_predictions = torch.empty(0).type(torch.BoolTensor)
        self.aleatoric_uncertainty = torch.empty(0).type(torch.FloatTensor)
        self.epistemic_uncertainty = torch.empty(0).type(torch.FloatTensor)
        self.total_uncertainty = torch.empty(0).type(torch.FloatTensor)

        self.model = ECGResNet_MCDropout_AuxOutput(in_length, in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation, n_dropout_samples, sampling_dropout_rate, n_logit_samples)


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
        x_i = self.model.sample_logits(self.n_logit_samples, output2_mean, output2_log_var, average=True)
            
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
        x_i = self.model.sample_logits(self.n_logit_samples, output2_mean, output2_log_var, average=True)
        
        # Apply softmax to obtain probability vector p_i
        p_i = F.softmax(x_i, dim=1)
        
        val_loss = self.loss(x_i, target)
        acc = FM.accuracy(p_i, target)

        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        metrics = {'val_loss': val_loss.item(), 'val_acc': acc.item()}
        self.log('val_acc', acc.item())
        self.log('val_loss', val_loss.item())
        return metrics

    def on_test_epoch_start(self):
        # Enable dropout at test time.
        self.model.enable_dropout()

    def test_step(self, batch, batch_idx, save_to_csv=False):
        data, target = batch['waveform'], batch['label']

        # MC sample using dropout, sample logits for every mc-dropout sample
        predictions, predictions_mean, predictions_var, log_variances_mean, predictions_mean_no_sm  = self.model.mc_sample_with_sample_logits(data)

        # Take exponent to get the variance
        output2_var = log_variances_mean.exp()

        predicted_labels = predictions_mean.argmax(dim=1)
        correct_predictions = torch.eq(predicted_labels, target)

        # MC dropout variance over predicted labels (epistemic uncertainty)
        sampled_var = torch.gather(predictions_var, 1, predictions_mean.argmax(dim=1).unsqueeze_(1))[:, 0]

        # Predicted aux-out variance of the predicted label (aleatoric uncertainty)
        predicted_labels_predicted_var = torch.gather(output2_var, 1, predictions_mean.argmax(dim=1).unsqueeze_(1))[:, 0]

        # Total uncertainty
        total_var = predicted_labels_predicted_var + sampled_var
        
        # Get metrics
        test_loss = self.loss(predictions_mean, target)
        acc = FM.accuracy(predictions_mean, target)

        self.log('test_acc', acc.item())
        self.log('test_loss', test_loss.item())

        self.IDs = torch.cat((self.IDs, batch['id']), 0)
        self.predicted_labels = torch.cat((self.predicted_labels, predicted_labels), 0)
        self.correct_predictions = torch.cat((self.correct_predictions, correct_predictions), 0)
        self.aleatoric_uncertainty = torch.cat((self.aleatoric_uncertainty, predicted_labels_predicted_var), 0)
        self.epistemic_uncertainty = torch.cat((self.epistemic_uncertainty, sampled_var), 0)
        self.total_uncertainty = torch.cat((self.total_uncertainty, total_var), 0)

        return {'test_loss': test_loss.item(), 'test_acc': acc.item()}

    # Initialize optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='none_auxout')
        parser.add_argument('--n_logit_samples', type=int, default=100) # Number of logit samples of the auxiliary output
        parser.add_argument('--n_dropout_samples', type=int, default=20)  # Number of dropout samples during MCDropout sampling
        parser.add_argument('--sampling_dropout_rate', type=float, default=0.1) # Dropout rate during MCDropout sampling
        parser.add_argument('--ensembling_method', type=bool, default=False)
        return parser

    # Combine results into single dataframe and save to disk
    def save_results(self):
        results = pd.concat([
            pd.DataFrame(self.IDs.numpy(), columns= ['ID']),  
            pd.DataFrame(self.predicted_labels.numpy(), columns= ['predicted_label']),
            pd.DataFrame(self.correct_predictions.numpy(), columns= ['correct_prediction']),
            pd.DataFrame(self.aleatoric_uncertainty.numpy(), columns= ['aleatoric_uncertainty']), 
            pd.DataFrame(self.epistemic_uncertainty.numpy(), columns= ['epistemic_uncertainty']), 
            pd.DataFrame(self.total_uncertainty.numpy(), columns= ['total_uncertainty']), 
        ], axis=1)

        create_results_directory()
        results.to_csv('results/{}_{}_results.csv'.format(self.__class__.__name__, datetime.datetime.now().replace(microsecond=0).isoformat()), index=False)
