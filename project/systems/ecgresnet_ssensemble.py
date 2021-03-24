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
from utils_.helpers import create_results_directory, create_weights_directory
from utils_.focalloss_weights import FocalLoss

class ECGResNetSnapshotEnsembleSystem(pl.LightningModule):
    """
    This class implements an snapshot ensemble of ECGResNets in PyTorch Lightning.
    It can estimate the epistemic uncertainty of its predictions.
    """
    def __init__(self, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, ensemble_size, max_epochs, initial_lr, cyclical_learning_rate_type, loss_weights=None, 
                 **kwargs):
        """
        Initializes the ECGResNetSnapshotEnsembleSystem

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
          max_epochs: total number of epochs to train for
          initial_lr: the initial learning rate at the start of a learning cycle
          cyclical_learning_rate_type: the type of learning rate cycling to apply
          loss_weights: array of weights for the loss term
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.ensemble_size = ensemble_size
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr
        self.cyclical_learning_rate_type = cyclical_learning_rate_type

        self.IDs = torch.empty(0).type(torch.LongTensor)
        self.predicted_labels = torch.empty(0).type(torch.LongTensor)
        self.correct_predictions = torch.empty(0).type(torch.BoolTensor)
        self.epistemic_uncertainty = torch.empty(0).type(torch.FloatTensor)

        self.models = []
        self.optimizers = []
        
        # Initialize a single model during training
        self.models.append(ECGResNet(in_channels, 
                           n_grps, N, num_classes, 
                           dropout, first_width, 
                           stride, dilation))

        if loss_weights is not None:
            weights = torch.tensor(loss_weights, dtype = torch.float)
        else:
            weights = loss_weights

        self.loss = FocalLoss(gamma=1, weights = weights)
        create_weights_directory()

    def forward(self, x, model_idx):
        """
        Performs a forward through a single ensemble member.

        Args:
            x (tensor): Input data.
            model_idx (int): Index of the ensemble member.

        Returns:
            Output1: Output at the auxiliary point of the ensemble member
            Output2: Output at the end of the ensemble member
        """
        output1, output2 = self.models[model_idx](x)
            
        return output1, output2

    def on_train_epoch_start(self):
        """
        Set the cyclical learning rate for the current epoch
        """
        learning_rate = self.get_learning_rate(self.current_epoch, self.ensemble_size, self.max_epochs, self.initial_lr, self.cyclical_learning_rate_type)
        self.set_learning_rate(self.optimizers[0], learning_rate)
        self.log('Learning rate', learning_rate)
        print('Epoch: {} learning rate: {}'.format(self.current_epoch, learning_rate))

    def training_step(self, batch, batch_idx):
        """Performs a training step for all ensemble members.

        Args:
            batch (dict): Output of the dataloader.
            batch_idx (int): Index no. of this batch.

        Returns:
            tensor: Total loss for this step.
        """
        data, target = batch['waveform'], batch['label']
        i = 0

        output1, output2 = self(data, 0)
        train_loss1 = self.loss(output1.squeeze(), target)
        train_loss2 = self.loss(output2.squeeze(), target)

        # Calculate the loss for single model
        total_train_loss = (0.3 * train_loss1) + train_loss2

        # Update weights for single model using optimizer
        self.manual_backward(total_train_loss, self.optimizers[i])
        self.optimizers[i].step()
        self.optimizers[i].zero_grad()

        self.log('model_{}_train_loss'.format(i), total_train_loss)

        return {'loss': total_train_loss}

    def on_train_epoch_end(self, outputs):
        """
        Save the model after each learning-rate cycle
        """
        if self.cyclical_learning_rate_type == 'cosine-annealing':
            epochs_per_cycle = self.max_epochs/self.ensemble_size

            # Check if we are at the end of a learning-rate cycle
            if (self.current_epoch +1) % epochs_per_cycle == 0:
                model_idx = int((self.current_epoch+1 )/ epochs_per_cycle)

                # Save current model 
                print('\nSaving model: {}/{}'.format(model_idx, self.ensemble_size))
                torch.save({
                        'epoch': self.current_epoch,
                        'model_state_dict': self.models[0].state_dict(),
                        'optimizer_state_dict': self.optimizers[0].state_dict(),
                },  "weights/ssensemble_model{}.pt".format(model_idx))
                # self.trainer.save_checkpoint("weights/ssensemble_model{}.ckpt".format(model_idx))

    def validation_step(self, batch, batch_idx):
        data, target = batch['waveform'], batch['label']

        # Always check the single model during validation
        i = 0

        # Predict using single model 
        output1, output2 = self(data, i)

        val_loss = self.loss(output2, target)
        acc = FM.accuracy(output2, target)

        # Log metrics
        metrics = {'val_loss': val_loss.item(), 'val_acc': acc.item()}
        self.log('val_acc', acc.item())
        self.log('val_loss', val_loss.item())
        return metrics

    def on_test_epoch_start(self):
        """
        Initialize ensemble members from saved checkpoints
        """
        print('\nInitializing ensemble members from checkpoints')

        # Remove first model from self.models
        self.models.clear()
        
        for i in range(self.ensemble_size):

            # Initialize ensemble members from different epochs in the training stage of the original model
            self.models.append(ECGResNet(self.hparams.in_channels, 
                           self.hparams.n_grps, self.hparams.N, self.hparams.num_classes, 
                           self.hparams.dropout, self.hparams.first_width, 
                           self.hparams.stride, self.hparams.dilation))

            model_path = 'weights/ssensemble_model{}.pt'.format(i+1)
            checkpoint = torch.load(model_path)
            self.models[i].load_state_dict(checkpoint['model_state_dict'])
            self.models[i].eval()

            print('Model {}/{} initialized\n'.format(i+1, self.ensemble_size))

    def test_step(self, batch, batch_idx, save_to_csv=False):
        prediction_individual = torch.empty(batch['label'].shape[0], self.ensemble_size, self.num_classes)
        data, target = batch['waveform'], batch['label']

        # Predict for each model in the ensemble
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
        Initialize the optimizer, during training only a single model is used
        """

        model_idx = 0
        self.optimizers.append(optim.SGD(self.models[model_idx].parameters(), lr=self.initial_lr))
        
        return self.optimizers

    def get_learning_rate(self, epoch_idx, n_models, total_epochs, initial_lr, cyclical_learning_rate_type):
        """
        Returns the learning rate for the current epoch.

        Args:
            epoch_idx: index of the current epoch
            n_models: total number of ensemble members
            total_epochs: total number of epochs to train for
            initial_lr: the initial learning rate at the start of a learning cycle
            cyclical_learning_rate_type: the type of learning rate cycling to apply
        """
        if cyclical_learning_rate_type == 'cosine-annealing':
            """
            Apply a cosine-annealing cyclical learning rate as proposed by
            Loshchilov et al. in: "SGDR: Stochastic Gradient Descent with Warm Restarts"
            """
            epochs_per_cycle = total_epochs/n_models 
            learning_rate = initial_lr * (np.cos(np.pi * (epoch_idx % epochs_per_cycle) / epochs_per_cycle) + 1) / 2
            return learning_rate
        else:
            return learning_rate
    
    def set_learning_rate(self, optimizer, learning_rate):
        """
        Sets the learning rate for an optimizer

        Args:
            optimizer: optimizer to apply learning rate to
            learning_rate: learning rate to set
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='ssensemble_none')
        parser.add_argument('--ensemble_size', type=int, default=2)
        parser.add_argument('--ensembling_method', type=bool, default=True)
        parser.add_argument('--initial_lr', type=float, default=0.1)
        parser.add_argument('--cyclical_learning_rate_type', type=str, default='cosine-annealing', choices=['cosine-annealing', 'none'])
        return parser

    # Combine results into single dataframe and save to disk
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
