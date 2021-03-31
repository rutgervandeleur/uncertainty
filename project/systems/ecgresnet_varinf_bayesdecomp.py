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

from network.ecgresnet_varinf import ECGResNet_VariationalInference, kldiv, get_beta, decompose_uncertainty
from utils_.helpers import create_results_directory
from utils_.focalloss_weights import FocalLoss

class ECGResNetVariationalInference_BayesianDecompositionSystem(pl.LightningModule):
    """
    This class implements the ECGResNet with Bayesian layers and Bayesian decomposition in PyTorch Lightning.
    It can estimate the epistemic and aleatoric uncertainty of its predictions.
    """
    def __init__(self, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, n_weight_samples,
                 kl_weighting_type, kl_weighting_scheme, max_epochs,
                 train_dataset_size, val_dataset_size, batch_size,
                 loss_weights=None, **kwargs):
        """
        Initializes the ECGResNetVariationalInference_BayesianDecompositionSystem

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
          n_weight_samples: number of Monte Carlo samples of the weights
          kl_weighting_type: which type of weighting to apply to the Kullback-Leibler term
          kl_weighting_scheme: which scheme of weighting to apply to the Kullback-Leibler term
          max_epochs: total number of epochs
          train_dataset_size: number of samples in the train dataset
          val_dataset_size: number of samples in the validation dataset
          batch_size: number of samples in a mini-batch
          loss_weights: array of weights for the loss term
        """
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
        self.predicted_labels = torch.empty(0).type(torch.LongTensor).to('cuda')
        self.correct_predictions = torch.empty(0).type(torch.BoolTensor).to('cuda')
        self.epistemic_uncertainty = torch.empty(0).type(torch.FloatTensor).to('cuda')
        self.aleatoric_uncertainty = torch.empty(0).type(torch.FloatTensor).to('cuda')
        self.total_uncertainty = torch.empty(0).type(torch.FloatTensor).to('cuda')

        self.model = ECGResNet_VariationalInference(in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation, n_weight_samples, kl_weighting_type, kl_weighting_scheme)

        if loss_weights is not None:
            weights = torch.tensor(loss_weights, dtype = torch.float)
        else:
            weights = loss_weights

        self.loss = FocalLoss(gamma=1, weights = weights)
        self.softmax = nn.Softmax(dim=1)

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
        acc = FM.accuracy(self.softmax(output2.squeeze()), target)

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
        epistemic_uncertainty, aleatoric_uncertainty = decompose_uncertainty(samples)

        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        # Get predicted labels by choosing the labels with the highest average Softmax value
        predicted_labels = sample_mean.argmax(dim=1)
        predicted_labels = predicted_labels.to('cuda')

        # Get the uncertainty of the predicted labels by selecting the uncertainty of the labels with highest average Softmax value
        predicted_labels_epi = torch.gather(epistemic_uncertainty, 1, sample_mean.argmax(dim=1).unsqueeze_(1))[:, 0]
        predicted_labels_ale = torch.gather(aleatoric_uncertainty, 1, sample_mean.argmax(dim=1).unsqueeze_(1))[:, 0]
        predicted_labels_total = torch.gather(total_uncertainty, 1, sample_mean.argmax(dim=1).unsqueeze_(1))[:, 0]

        predicted_labels_epi = predicted_labels_epi.to('cuda')
        predicted_labels_ale = predicted_labels_ale.to('cuda')
        predicted_labels_total = predicted_labels_total.to('cuda')


        correct_predictions = torch.eq(predicted_labels, target)

        # Get metrics
        test_loss = self.loss(sample_mean, target)
        acc = FM.accuracy(sample_mean.to('cuda'), target)

        self.log('test_acc', acc.item())
        self.log('test_loss', test_loss.item())
    
        #self.IDs = torch.cat((self.IDs,batch['id']), 0)
        self.predicted_labels = torch.cat((self.predicted_labels, predicted_labels), 0)
        self.correct_predictions = torch.cat((self.correct_predictions, correct_predictions), 0)
        self.epistemic_uncertainty = torch.cat((self.epistemic_uncertainty, predicted_labels_epi), 0)
        self.aleatoric_uncertainty = torch.cat((self.aleatoric_uncertainty, predicted_labels_ale), 0)
        self.total_uncertainty = torch.cat((self.total_uncertainty, predicted_labels_total), 0)

        return {'test_loss': test_loss.item(), 'test_acc': acc.item(), 'test_loss': test_loss.item()}

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
        """
        Combine results into single dataframe and save to disk as .csv file
        """
        results = pd.concat([
            pd.DataFrame(self.IDs.numpy(), columns= ['ID']),  
            pd.DataFrame(self.predicted_labels.cpu().numpy(), columns= ['predicted_label']),
            pd.DataFrame(self.correct_predictions.cpu().numpy(), columns= ['correct_prediction']),
            pd.DataFrame(self.epistemic_uncertainty.cpu().numpy(), columns= ['epistemic_uncertainty']), 
            pd.DataFrame(self.aleatoric_uncertainty.cpu().numpy(), columns= ['aleatoric_uncertainty']), 
            pd.DataFrame(self.total_uncertainty.cpu().numpy(), columns= ['total_uncertainty']), 
        ], axis=1)

        create_results_directory()
        results.to_csv('results/{}_{}_results.csv'.format(self.__class__.__name__, datetime.datetime.now().replace(microsecond=0).isoformat()), index=False)
