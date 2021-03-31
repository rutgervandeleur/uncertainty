from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import datetime
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

from network.ecgresnet import BasicBlock, Flatten

class ECGResNet_MCDropout_AuxOutput(nn.Module):
    """
    This class implements the ECG-ResNet with Monte-Carlo dropout and Auxiliary output in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ResNet object can perform forward.
    The Monte-Carlo dropout samples are used to estimate the epistemic uncertainty.
    The Auxiliary output is used to estimate the aleatoric uncertainty.
    """
    def __init__(self, in_channels, n_grps, N, num_classes, dropout, first_width, 
                 stride, dilation, n_dropout_samples, sampling_dropout_rate, n_logit_samples, train=False):
        """
        Initializes ECGResNet_MCDropout_AuxOutput object. 

        Args:
          in_channels: number of channels of input
          n_grps: number of ResNet groups
          N: number of blocks per groups
          num_classes: number of classes of the classification problem
          dropout: probability of an argument to get zeroed in the dropout layer
          first_width: width of the first input
          stride: tuple with stride value per block per group
          dilation: spacing between the kernel points of the convolutional layers
          n_dropout_samples: number of dropout samples to take
          sampling_dropout_rate: the ratio of dropped-out neurons during MC sampling
          n_logit_samples: number of logit samples to take of the auxiliary output
        """
        super().__init__()
        num_branches = 2
        first_width = first_width * num_branches
        stem = [nn.Conv1d(in_channels, first_width // 2, kernel_size=7, padding=3, 
                          stride = 2, dilation = 1, bias=False),
                nn.BatchNorm1d(first_width // 2), nn.ReLU(),
                nn.Conv1d(first_width // 2, first_width, kernel_size = 1, 
                          padding = 0, stride = 1,  bias = False),
                nn.BatchNorm1d(first_width), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(first_width, first_width, kernel_size = 5, 
                          padding = 2, stride = 1, bias = False)]
        
        layers = []
        
        # Double feature depth at each group, after the first
        widths = [first_width]
        for grp in range(n_grps):
            widths.append((first_width)*2**grp)
        for grp in range(n_grps):
            layers += self._make_group(N, widths[grp], widths[grp+1],
                                       stride, dropout, dilation, num_branches)
        
        layers += [nn.BatchNorm1d(widths[-1]), nn.ReLU(inplace=True)]
        fclayers1 = [nn.Linear(20096, 256), nn.ReLU(inplace = True), 
                    nn.Dropout(dropout), nn.Linear(256, num_classes)]
        fclayers2 = [nn.Linear(5120, 256), nn.ReLU(inplace = True), 
                    nn.Dropout(dropout), nn.Linear(256, 2*num_classes)]
        
        self.stem = nn.Sequential(*stem)
        aux_point = (len(layers) - 2) // 2
        self.features1 = nn.Sequential(*layers[:aux_point])
        self.features2 = nn.Sequential(*layers[aux_point:])
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(*fclayers1)
        self.fc2 = nn.Sequential(*fclayers2)
        self.num_classes = num_classes
        self.n_dropout_samples = n_dropout_samples
        self.sampling_dropout_rate = sampling_dropout_rate # Dropout during MCDropout sampling
        self.n_logit_samples = n_logit_samples # Number of logit samples of the auxiliary output
        self.Gauss = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(num_classes), torch.eye(num_classes))

    def _make_group(self, N, in_channels, out_channels, stride, dropout, dilation, num_branches):
        """
        Builds a group of blocks.

        Args:
          N: number of blocks per groups
          in_channels: number of channels of input
          out_channels: number of channels of output
          stride: stride of convolutions
          dropout: probability of an argument to get zeroed in the dropout layer
          dilation: spacing between the kernel points of the convolutional layers
          num_branches: number of branches of the block
        """
        group = list()
        for i in range(N):
            blk = BasicBlock(in_channels=(in_channels if i == 0 else out_channels), 
                             out_channels=out_channels, stride=stride[i], 
                             dropout = dropout, dilation = dilation, 
                             num_branches = num_branches)
            group.append(blk)
        return group
 
    def forward(self, x):
        x = self.stem(x)
        x1 = self.features1(x)
        x1out = self.flatten(x1)
        x2 = self.features2(x1)
        x2out = self.flatten(x2)
        
        # Get logits
        logits = self.fc2(x2out)

        # Split into mean and log_variance
        output2_mean = logits[:, 0:self.num_classes]
        output2_log_var = logits[:, self.num_classes:]
        return self.fc1(x1out), output2_mean, output2_log_var

   
    # Turn on the dropout layers
    def enable_dropout(self):
        """
        Turns on the dropout layers, sets the dropout rate to self.sampling_dropout_rate
        """
        for module in self.modules():
            if module.__class__.__name__.startswith('Dropout'):
                # Turn on dropout
                module.train()
                
                # Set dropout rate
                module.p = self.sampling_dropout_rate
        
    # Takes n Monte Carlo samples 
    def mc_sample_with_sample_logits(self, data):
        """
        Takes Monte Carlo dropout samples of the network by repeatedly
        applying a dropout mask and making a prediction using that mask.
        For each MC dropout sample, the logits of the network are sampled n_logit_samples times
        to obtain the aleatoric uncertainty of the prediction.

        Args:
            data: data point to forward
        """
        predictions = torch.empty((data.shape[0], self.n_dropout_samples, self.num_classes))
        predictions_no_sm = torch.empty((data.shape[0], self.n_dropout_samples, self.num_classes))
        log_variances = torch.empty((data.shape[0], self.n_dropout_samples, self.num_classes))
        
        for i in range(self.n_dropout_samples):
            # forward push
            _, output2_mean, output2_log_var = self(data)
            
            # Sample from logits, returning a vector x_i
            x_i = self.sample_logits(self.n_logit_samples, output2_mean, output2_log_var, average=True)
            
            # Apply softmax to obtain probability vector p_i
            p_i = F.softmax(x_i, dim=1)
            
            # Save results
            predictions[:, i] = p_i
            predictions_no_sm[:, i] = x_i
            log_variances[:, i] = output2_log_var
        
        # Calculate mean and variance over the predictions, mean over log_variances, return results
        predictions_mean = predictions.mean(dim=1)
        predictions_mean_no_sm = predictions_no_sm.mean(dim=1)
        predictions_var = predictions.var(dim=1)
        log_variances_mean = log_variances.mean(dim=1)
        
        return predictions, predictions_mean, predictions_var, log_variances_mean, predictions_mean_no_sm 

    def sample_logits(self, T, input, log_var, average=True, device='cpu'):
        """
        Takes T samples from the logits, by corrupting the network output with
        Gaussian noise with variance determined by the networks auxiliary
        outputs. 
        As in "What uncertainties do we need in Bayesian deep learning for
        computer vision?", equation (12), first part. "In practice, we train 
        the network to predict the log variance instead of the normal variance."

        Args:
            T: number of logits samples
            log_var: the log variance as predicted by the auxiliary output
            average: whether to average the result
        """
        
        # Take the exponent to get the variance
        variance = log_var.exp()

        # Go from shape: [batch x num_classes] -> [batch x T x num_classes]
        sigma = variance[:, None, :].repeat(1, T, 1)
        f = input[:, None, :].repeat(1, T, 1)

        # Take T samples from the Gaussian distribution
        epsilon = self.Gauss.sample([input.shape[0], T]).to(device)

        # Multiply Gaussian noise with variance, and add to the prediction
        x_i = f + (sigma * epsilon) 

        if average==True:
            return x_i.mean(dim=1)
        else:
            return x_i

