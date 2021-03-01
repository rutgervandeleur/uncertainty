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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import numpy as np
import pandas as pd

from utils.helpers import convert_predictions_to_expert_categories, convert_variances_to_expert_categories
from network.ecgresnet import BasicBlock, Flatten

class ECGResNet_AuxOut(nn.Module):
    """
    This class implements the ECG-ResNet with auxiliary output in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ResNet object can perform forward.
    The auxiliary output allows for the estimation of aleatoric uncertainty
    which is learned from the data.
    """
    def __init__(self, in_channels, n_grps, N, num_classes, dropout, first_width, 
                 stride, dilation):
        """
        Initializes ECGResNet object with Auxiliary output for aleatoric uncertainty estimation. 

        Args:
          in_channels: number of channels of input
          n_grps: number of ResNet groups
          N: number of blocks per groups
          num_classes: number of classes of the classification problem
          dropout: probability of an argument to get zeroed in the dropout layer
          first_width: width of the first input
          stride: tuple with stride value per block per group
          dilation: spacing between the kernel points of the convolutional layers
        """
        super().__init__()
        self.dropout = dropout # Dropout during training
        self.softmax = nn.Softmax(dim=1)
        self.num_classes = num_classes
        self.Gauss = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(num_classes), torch.eye(num_classes))

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

    def sample_logits(self, T, input, log_var, average=True):
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
        epsilon = self.Gauss.sample([input.shape[0], T])

        # Multiply Gaussian noise with variance, and add to the prediction
        x_i = f + (sigma * epsilon) 

        if average==True:
            return x_i.mean(dim=1)
        else:
            return x_i

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
