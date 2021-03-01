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

class ECGResNet_MCDropout(nn.Module):
    """
    This class implements the ECG-ResNet in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ResNet object can perform forward.
    """
    def __init__(self, in_channels, n_grps, N, num_classes, dropout, first_width, 
                 stride, dilation, n_dropout_samples, sampling_dropout_rate):
        """
        Initializes ECGResNet object. 

        Args:
          in_channels: number of channels of input
          n_grps: number of ResNet groups
          N: number of blocks per groups
          num_classes: number of classes of the classification problem
          stride: tuple with stride value per block per group
        """
        super().__init__()
        self.dropout = dropout # Dropout during training
        self.softmax = nn.Softmax(dim=1)
        self.num_classes = num_classes
        self.n_samples = n_dropout_samples
        self.sampling_dropout_rate = sampling_dropout_rate # Dropout during MC sampling

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
                    nn.Dropout(dropout), nn.Linear(256, num_classes)]
        
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
          in_channels: number of channels of input
          out_channels: number of channels of output
          stride: stride of convolutions
          N: number of blocks per groups
          num_classes: number of classes of the classification problem
        """
        group = list()
        for i in range(N):
            blk = BasicBlock(in_channels=(in_channels if i == 0 else out_channels), 
                             out_channels=out_channels, stride=stride[i], 
                             dropout = dropout, dilation = dilation, 
                             num_branches = num_branches)
            group.append(blk)
        return group

    # Turn on the dropout layers
    def enable_dropout(self):
        for module in self.modules():
            if module.__class__.__name__.startswith('Dropout'):
                # Turn on dropout
                module.train()
                
                # Set dropout rate
                module.p = self.sampling_dropout_rate
        
    # Takes n Monte Carlo samples 
    def mc_sample(self, data):
        samples = torch.empty((data.shape[0], self.n_samples, self.num_classes))
        
        for i in range(self.n_samples):
            # forward push
            _, output2 = self(data)
            predictions = self.softmax(output2)

            # Save results
            samples[:, i] = predictions
        
        # Calculate mean and variance over the samples, return results
        sample_mean = samples.mean(dim=1)
        sample_var = samples.var(dim=1)
        
        return samples, sample_mean, sample_var

    def forward(self, x):
        x = self.stem(x)
        x1 = self.features1(x)
        x1out = self.flatten(x1)
        x2 = self.features2(x1)
        x2out = self.flatten(x2)
        return self.fc1(x1out), self.fc2(x2out)


