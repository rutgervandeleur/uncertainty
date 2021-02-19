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

from uncertainty.network.ecgresnet_varinf import ECGResNet_VariationalInference, kldiv, get_beta, decompose_uncertainty
from ecgnet.utils.loss import SoftmaxFocalLoss
from uncertainty.utils.helpers import create_results_directory
from uncertainty.utils.focalloss_weights import FocalLoss

class ECGResNetVariationalInferenceSystem(pl.LightningModule):

    def __init__(self, in_length, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, n_weight_samples, kl_weighting_type, kl_weighting_scheme, max_epochs, train_dataset_size, val_dataset_size, batch_size, loss_weights=None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.n_weight_samples = n_weight_samples
        self.kl_weighting_type = kl_weighting_type
        self.kl_weighting_scheme = kl_weighting_scheme
        self.max_epochs = max_epochs
        self.train_dataset_size = train_dataset_size
        self.val_dataset_size = val_dataset_size
        self.batch_size = batch_size 

        self.IDs = torch.empty(0).type(torch.LongTensor)
        self.predicted_labels = torch.empty(0).type(torch.LongTensor)
        self.correct_predictions = torch.empty(0).type(torch.BoolTensor)
        self.epistemic_uncertainty = torch.empty(0).type(torch.FloatTensor)
        self.aleatoric_uncertainty = torch.empty(0).type(torch.FloatTensor)
        self.total_uncertainty = torch.empty(0).type(torch.FloatTensor)

        self.model = ECGResNet_VariationalInference(in_length, in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation, n_weight_samples, kl_weighting_type, kl_weighting_scheme)

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

        # Calculate Focal loss for mid and final output
        train_loss1 = self.loss(output1, target)
        train_loss2 = self.loss(output2, target)

        # Calculate kl divergence over all Bayesian layers
        kl_clean = kldiv(self.model) 
    
        # Weight the KL divergence, so it does not overflow the loss term
        kl = self.model.weight_kl(kl_clean, self.train_dataset_size)
        
        # Apply KL weighting scheme, allows for balancing the KL term non-uniformly
        M = self.train_dataset_size / self.batch_size
        beta = get_beta(batch_idx, M, beta_type=self.kl_weighting_scheme, epoch=self.current_epoch+1, num_epochs=self.max_epochs)
        kl_weighted = beta * kl
        
        # Variational inference objective = -Kl divergence + negative log likelihood
        ELBO = kl_weighted + train_loss2

        # Calculate total loss
        total_train_loss = (0.3 * train_loss1) + ELBO
    
        self.log('train_loss', total_train_loss)
        self.log('train_ELBO', ELBO)
        self.log('train_kl_weighted', kl_weighted)

        return {'loss': total_train_loss}
    
    def validation_step(self, batch, batch_idx):
        data, target = batch['waveform'], batch['label']

        # Perform step
        _, output2 = self(data)
        
        # Calculate loss, must be CrossEntropy or a derivative
        val_loss = self.loss(output2, target)
        
        # Calculate KL divergence between the approximate posterior and the prior over all Bayesian layers
        kl_clean = kldiv(self.model) 

        # Weight the KL divergence, so it does not overflow the loss term
        kl = self.model.weight_kl(kl_clean, self.val_dataset_size)      

        # Apply KL weighting scheme, allows for balancing the KL term non-uniformly
        M = self.val_dataset_size / self.batch_size
        beta = get_beta(batch_idx, M, beta_type=self.kl_weighting_scheme)
        kl_weighted = beta * kl

        # Calculate accuracy
        acc = FM.accuracy(output2.squeeze(), target)

        # Loss is tensor
        metrics = {'val_loss': val_loss.item(), 'val_acc': acc.item()}
        self.log('val_acc', acc.item())
        self.log('val_loss', val_loss.item())
        self.log('val_kl_weighted', kl_weighted.item())

        return metrics

    def test_step(self, batch, batch_idx, save_to_csv=False):
        data, target = batch['waveform'], batch['label']

        # Sample the weights and use sample to make prediction
        samples, sample_mean, sample_var, samples_no_sm, sample_mean_no_sm = self.model.sample_weights(data)

        # Decompose predictive uncertainty into epistemic and aleatoric uncertainty
        epistemic_uncertainty, aleatoric_uncertainty = decompose_uncertainty(samples, self.n_weight_samples)

        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        # Get predicted labels by choosing the labels with the highest average Softmax value
        predicted_labels = sample_mean.argmax(dim=1)

        # Get the uncertainty of the predicted labels by selecting the uncertainty of the labels with highest average Softmax value
        predicted_labels_epi = torch.gather(epistemic_uncertainty, 1, sample_mean.argmax(dim=1).unsqueeze_(1))[:, 0]
        predicted_labels_ale = torch.gather(aleatoric_uncertainty, 1, sample_mean.argmax(dim=1).unsqueeze_(1))[:, 0]
        predicted_labels_total = torch.gather(total_uncertainty, 1, sample_mean.argmax(dim=1).unsqueeze_(1))[:, 0]

        correct_predictions = torch.eq(predicted_labels, target)

        # Get metrics
        test_loss = self.loss(sample_mean, target)
        acc = FM.accuracy(sample_mean, target)

        self.log('test_acc', acc.item())
        self.log('test_loss', test_loss.item())

        self.IDs = torch.cat((self.IDs, batch['id']), 0)
        self.predicted_labels = torch.cat((self.predicted_labels, predicted_labels), 0)
        self.correct_predictions = torch.cat((self.correct_predictions, correct_predictions), 0)
        self.epistemic_uncertainty = torch.cat((self.epistemic_uncertainty, predicted_labels_epi), 0)
        self.aleatoric_uncertainty = torch.cat((self.aleatoric_uncertainty, predicted_labels_ale), 0)
        self.total_uncertainty = torch.cat((self.total_uncertainty, predicted_labels_total), 0)

        return {'test_loss': test_loss.item(), 'test_acc': acc.item(), 'test_loss': test_loss.item()}

    # Initialize optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='mcdropout_none')
        parser.add_argument('--n_weight_samples', type=int, default=25)
        parser.add_argument('--kl_weighting_type', type=str, default='parameter_size', choices=['dataset_size', 'parameter_size'])
        parser.add_argument('--kl_weighting_scheme', type=str, default='Standard', choices=['Standard', 'Blundell', 'Soenderby'])
        parser.add_argument('--ensembling_method', type=bool, default=False)
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
