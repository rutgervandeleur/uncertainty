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
from utils.helpers import create_results_directory, create_weights_directory
from utils.focalloss_weights import FocalLoss

class ECGResNetSnapshotEnsemble_AuxOutSystem(pl.LightningModule):
    """
    This class implements an snapshot ensemble of ECGResNets with auxiliary output in PyTorch Lightning.
    It can estimate the epistemic and aleatoric uncertainty of its predictions.
    """
    def __init__(self, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, learning_rate, ensemble_size, max_epochs, initial_lr, cyclical_learning_rate_type, n_logit_samples, loss_weights=None, 
                 **kwargs):
        """
        Initializes the ECGResNetSnapshotEnsemble_AuxOutSystem

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
          n_logit_samples: number of logit samples of the auxiliary output
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
        self.n_logit_samples = n_logit_samples

        self.IDs = torch.empty(0).type(torch.LongTensor)
        self.predicted_labels = torch.empty(0).type(torch.LongTensor)
        self.correct_predictions = torch.empty(0).type(torch.BoolTensor)
        self.epistemic_uncertainty = torch.empty(0).type(torch.FloatTensor)
        self.aleatoric_uncertainty = torch.empty(0).type(torch.FloatTensor)
        self.total_uncertainty = torch.empty(0).type(torch.FloatTensor)

        self.models = []
        self.optimizers = []

        # Device needs to be selected because PyTorch Lightning does not
        # recognize multiple models when in list
        manual_device = torch.device('cuda' if torch.cuda.is_available() and kwargs['gpus'] != 0 else 'cpu')

        self.models.append(ECGResNet_AuxOut(in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation).to(manual_device)
                              )
        if loss_weights is not None:
            weights = torch.tensor(loss_weights, dtype = torch.float)
        else:
            weights = loss_weights

        self.loss = FocalLoss(gamma=1, weights = weights)
        create_weights_directory()

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
        model_idx = 0

        # Make prediction
        output1, output2_mean, output2_log_var = self(data, model_idx)

        # Sample from logits, returning a vector x_i
        x_i = self.models[model_idx].sample_logits(self.n_logit_samples, output2_mean, output2_log_var, average=True)

        train_loss1 = self.loss(output1, target)
        train_loss2 = self.loss(x_i, target)
        total_train_loss = (0.3 * train_loss1) + train_loss2

        # Update weights for single model using individual optimizer
        self.manual_backward(total_train_loss, self.optimizers[model_idx])
        self.optimizers[model_idx].step()
        self.optimizers[model_idx].zero_grad()

        self.log('train_loss'.format(model_idx), total_train_loss)

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
                },  "weights/ssensemble_auxout_model{}.pt".format(model_idx))
                # self.trainer.save_checkpoint("weights/ssensemble_model{}.ckpt".format(model_idx))

    def validation_step(self, batch, batch_idx):
        data, target = batch['waveform'], batch['label']
        model_idx = 0

        # Make prediction
        _, output2_mean, output2_log_var = self(data, model_idx)

        # Sample from logits, returning vector x_i
        x_i = self.models[model_idx].sample_logits(self.n_logit_samples, output2_mean, output2_log_var, average=True)

        # Apply softmax to obtain probability vector p_i
        p_i = F.softmax(x_i, dim=1)
        
        val_loss = self.loss(p_i, target)
        acc = FM.accuracy(p_i, target)

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
        
        for model_idx in range(self.ensemble_size):

            # Initialize ensemble members from different epochs in the training stage of the original model
            self.models.append(ECGResNet_AuxOut(self.hparams.in_channels, 
                               self.hparams.n_grps, self.hparams.N, self.hparams.num_classes, 
                               self.hparams.dropout, self.hparams.first_width, 
                               self.hparams.stride, self.hparams.dilation, self.hparams.n_logit_samples)
                              )


            model_path = 'weights/ssensemble_auxout_model{}.pt'.format(model_idx+1)
            checkpoint = torch.load(model_path)
            self.models[model_idx].load_state_dict(checkpoint['model_state_dict'])
            self.models[model_idx].eval()

            print('Model {}/{} initialized\n'.format(model_idx+1, self.ensemble_size))

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
        parser.add_argument('--n_logit_samples', type=int, default=100)
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
            pd.DataFrame(self.aleatoric_uncertainty.numpy(), columns= ['aleatoric_uncertainty']), 
            pd.DataFrame(self.total_uncertainty.numpy(), columns= ['total_uncertainty']), 
        ], axis=1)

        create_results_directory()
        results.to_csv('results/{}_{}_results.csv'.format(self.__class__.__name__, datetime.datetime.now().replace(microsecond=0).isoformat()), index=False)
