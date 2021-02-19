from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torchnet.logger import VisdomPlotLogger, VisdomLogger

import time
import sys
import datetime
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
import torchnet as tnt
from torchnet.engine import Engine
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import numpy as np
import pandas as pd

from .ecgresnet import *
from uncertainty.utils.helpers import convert_predictions_to_expert_categories, convert_variances_to_expert_categories

class BayesLinear(nn.Module):
    """
    This class implements a Bayesian Linear layer.
    """
    def __init__(self, in_features, out_features, bias=True, log_sigma_prior=-5,
                 mu_prior=-1):
        """
        Initializes BayesLinear layer. 

        Args:
            
           """
        super(BayesLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Initialize the parameters mu and sigma which will aproximate the Bayesian posterior
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Model the log sigma, so the variance will always be positive
        self.w_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initial mu prior
        #self.mu_prior = torch.Tensor(out_features, in_features)
        self.mu_prior_init = mu_prior

        # Initial log sigma prior
        #self.log_sigma_prior = torch.Tensor(out_features, in_features)
        #self.log_sigma_prior_init = log_sigma_prior
        self.log_sigma_prior_init = log_sigma_prior

        # Initialize the log variance uniformly, the exponent will be around 0
        #init.kaiming_uniform_(self.mu_prior, a=math.sqrt(5))
        #init.uniform_(self.log_sigma_prior, self.log_sigma_prior_init-0.1, self.log_sigma_prior_init)
                
        if bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()


    def reset_parameters(self):
        # ReLU activations are used, so init using Kaiming initialisation
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        
        # Initialize the log variance uniformly, the exponent will be around 0
        init.uniform_(self.w_log_sigma, self.log_sigma_prior_init-0.1, self.log_sigma_prior_init)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Performs forward pass of the input.

        Args:
          x: 
        Returns:
          out: 
        """

        # Calculate activations
        act_mu = F.linear(input, self.w_mu, self.bias)
        act_sigma = torch.sqrt(F.linear(input ** 2, torch.exp(self.w_log_sigma) ** 2) + 1e-8)
        
        # Sample from unit gaussian
        epsilon = torch.randn_like(act_mu)

        # Apply reparameterization trick to sample from activations
        return act_mu + act_sigma * epsilon
        
    def kl(self):
        return calculate_kl(torch.Tensor([self.mu_prior_init]), torch.exp(torch.Tensor([self.log_sigma_prior_init])),
                            self.w_mu, torch.exp(self.w_log_sigma))

class BayesConv1d(nn.Module):
    """
    This class implements a Bayesian 1-dimensional Convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, bias=True, log_sigma_prior=-5,
                 mu_prior=-1):
        """
        Initializes BayesLinear layer. 

        Args:
            
           """
        super(BayesConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize the parameters 
        self.w_mu = nn.Parameter(torch.Tensor(out_channels, 
                                              in_channels,
                                              kernel_size))
        
        # Model the log sigma, so variance will always be positive
        self.w_log_sigma = nn.Parameter(torch.Tensor(out_channels,
                                                      in_channels,
                                                      kernel_size))
        
        #self.mu_prior  = torch.Tensor(out_channels,
        #                              in_channels,
        #                              kernel_size)
        
        # Initial mu prior
        self.mu_prior_init = mu_prior

        #self.log_sigma_prior  = torch.Tensor(out_channels,
        #                              in_channels,
        #                              kernel_size)

        # Initial log sigma (variance) prior
        self.log_sigma_prior_init = log_sigma_prior

        #init.kaiming_uniform_(self.mu_prior, a=math.sqrt(5))
        #init.uniform_(self.log_sigma_prior, self.log_sigma_prior_init-0.1, self.log_sigma_prior_init)
                
        if bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        
        # Initialize the log variance uniformly, the exponent will be around 0
        init.uniform_(self.w_log_sigma, self.log_sigma_prior_init-0.1, self.log_sigma_prior_init)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Performs forward pass of the input.

        Args:
          x: 
        Returns:
          out: 
        """

        # Calculate activations for mu and sigma
        act_mu = F.conv1d(input, self.w_mu, self.bias, self.stride, self.padding,
                          self.dilation)
        
        # Use the clamp to make sure no negative values pass through to the square root
        act_sigma = torch.sqrt( torch.clamp(F.conv1d(input ** 2, torch.exp(self.w_log_sigma) ** 2, self.bias, self.stride, self.padding, self.dilation), min=1e-16))

        # Sample from unit gaussian
        epsilon = torch.randn_like(act_mu)

        # Apply reparameterization trick to sample from activations
        return act_mu + act_sigma * epsilon
        
    
    def kl(self):
        return calculate_kl(torch.Tensor([self.mu_prior_init]), torch.exp(torch.Tensor([self.log_sigma_prior_init])),
                            self.w_mu, torch.exp(self.w_log_sigma))

class BayesBasicBlock(nn.Module):
    """
    This class implements a Bayesian residual block.
    """
    def __init__(self, in_channels, out_channels, stride, dropout, dilation, num_branches):
        """
        Initializes Bayesian BasicBlock object. 

        Args:
          in_channels: number of input channels
          out_channels: number of output channels
          stride: stride of the convolution
          dropout: probability of an argument to get zeroed in the dropout layer
        """
        super(BayesBasicBlock, self).__init__()
        kernel_size = 5
        
        self.branch0 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace = True),
            BayesConv1d(in_channels, in_channels // num_branches, kernel_size = 1, 
                      padding = 0, stride = 1, dilation = 1,  bias = False),

            nn.BatchNorm1d(in_channels // num_branches),
            nn.ReLU(),
            nn.Dropout(dropout),
            BayesConv1d(in_channels // num_branches, out_channels // num_branches, kernel_size = kernel_size, 
                      padding = (kernel_size - 1) // 2, stride = stride, dilation = 1, bias = False)
        )
        
        self.branch1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace = True),
            BayesConv1d(in_channels, in_channels // num_branches, kernel_size = 1, 
                      padding = 0, stride = 1, dilation = 1, bias = False),

            nn.BatchNorm1d(in_channels // num_branches),
            nn.ReLU(),
            nn.Dropout(dropout),
            BayesConv1d(in_channels // num_branches, out_channels // num_branches, kernel_size = kernel_size, 
                      padding = ((kernel_size - 1) * dilation) // 2, stride = stride, 
                      dilation = dilation, bias = False)     
        )
        
        if in_channels == out_channels and stride == 1:
            self.shortcut = lambda x: x
            
        else:
            self.shortcut = BayesConv1d(in_channels, out_channels, kernel_size = 1, 
                                      padding = 0, stride = stride, dilation = 1, bias=False)
            

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through 
        several layer transformations.

        Args:
          x: input to the block with size NxCxL
        Returns:
          out: outputs of the block
        """
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0,x1), 1)
        r = self.shortcut(x)
        return out.add_(r)

class ECGResNet_VariationalInference(nn.Module):
    """
    This class implements the ECG-ResNet using variational inference in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ResNet object can perform forward.
    """
    def __init__(self, in_length, in_channels, n_grps, N, num_classes, dropout, first_width, 
                 stride, dilation, n_weight_samples, kl_weighting_type, kl_weighting_scheme, train=False):
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
        self.dropout = dropout
        self.softmax = nn.Softmax(dim=1)
        self.num_classes = num_classes

        self.n_weight_samples = n_weight_samples
        self.kl_weighting_type  = kl_weighting_type
        self.kl_weighting_scheme  = kl_weighting_scheme
        # self.WeightedCrossEntropyLoss = nn.CrossEntropyLoss(weight=torch.Tensor(train_params['loss_weights']))

        num_branches = 2
        first_width = first_width * num_branches
        stem = [BayesConv1d(in_channels, first_width // 2, kernel_size=7, padding=3, 
                          stride = 2, dilation = 1, bias=False),
                nn.BatchNorm1d(first_width // 2), nn.ReLU(),
                BayesConv1d(first_width // 2, first_width, kernel_size = 1, 
                          padding = 0, stride = 1, dilation = 1,  bias = False),
                nn.BatchNorm1d(first_width), nn.ReLU(), nn.Dropout(dropout),
                BayesConv1d(first_width, first_width, kernel_size = 5, 
                          padding = 2, stride = 1, dilation = 1, bias = False)]
        
        layers = []
        
        # Double feature depth at each group, after the first
        widths = [first_width]
        for grp in range(n_grps):
            widths.append((first_width)*2**grp)
        for grp in range(n_grps):
            layers += self._make_group(N, widths[grp], widths[grp+1],
                                       stride, dropout, dilation, num_branches)
        
        layers += [nn.BatchNorm1d(widths[-1]), nn.ReLU(inplace=True)]
        fclayers1 = [BayesLinear(20096, 256), nn.ReLU(inplace = True), 
                    nn.Dropout(dropout), BayesLinear(256, num_classes)]
        fclayers2 = [BayesLinear(5120, 256), nn.ReLU(inplace = True), 
                    nn.Dropout(dropout), BayesLinear(256, num_classes)]
        
        self.stem = nn.Sequential(*stem)
        aux_point = (len(layers) - 2) // 2
        self.features1 = nn.Sequential(*layers[:aux_point])
        self.features2 = nn.Sequential(*layers[aux_point:])
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(*fclayers1)
        self.fc2 = nn.Sequential(*fclayers2)
        
        self.criterion = F.cross_entropy
        
        # Total number of parameters of which a KL divergence can be calculated:
        self.num_kl_params = get_num_kl_parameters(self)
        
        # Count the amount of trainable parameters
        self.num_train_params = count_parameters(self)
       
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
            blk = BayesBasicBlock(in_channels=(in_channels if i == 0 else out_channels), 
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
        return self.fc1(x1out), self.fc2(x2out)

    # Takes n Monte Carlo samples of the weights 
    def sample_weights(self, data):
        samples = torch.empty((data.shape[0], self.n_weight_samples, self.num_classes))
        samples_no_sm = torch.empty((data.shape[0], self.n_weight_samples, self.num_classes))
        
        for i in range(self.n_weight_samples):
            # forward push
            _, output2 = self(data)
            predictions = F.softmax(output2, dim=1)
            
            # Save results
            samples[:, i] = predictions
            samples_no_sm[:, i] = output2
        
        # Calculate mean and variance over the samples, return results
        sample_mean = samples.mean(dim=1)
        sample_mean_no_sm = samples_no_sm.mean(dim=1)
        sample_var = samples.var(dim=1)
        
        return samples, sample_mean, sample_var, samples_no_sm, sample_mean_no_sm

    # Weights the KL term according to the weighting type
    def weight_kl(self, kl_clean, dataset_size):
        
        if self.kl_weighting_type == 'dataset_size':
            kl = kl_clean / dataset_size
            
        elif self.kl_weighting_type == 'parameter_size':
            kl = kl_clean / self.num_train_params
            
        return kl

    def kl(self):
        return calculate_kl(torch.Tensor([self.mu_prior_init]), torch.exp(torch.Tensor([self.log_sigma_prior_init])).to(device),self.w_mu, torch.exp(self.w_log_sigma))

# kl-divergence between to univariate gaussians (p = prior, q is approximated posterior)
def calculate_kl(mu_p, sig_p, mu_q, sig_q):
    kl = 0.5 * (2 * torch.log(sig_p/ sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

def kldiv(model):
    kl = torch.Tensor([0.0])
    for c in model.children():
        if hasattr(c, 'children'):
            kl += kldiv(c)
        if hasattr(c, 'kl'):
            kl += c.kl()
    return kl

# Returns the number of neurons that contribute to the kl divergence
def get_num_kl_parameters(model):
    count = 0
    for c in model.children():
        if hasattr(c, 'children'):
            count += get_num_kl_parameters(c)
        if hasattr(c, 'kl'):
            count += 1
    return count

# Returns the number of train
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Decompose the predictive uncertainty into an epistemic and aleatoric part.
# Technique described in Y. Kwon et al
# Taken from https://openreview.net/pdf?id=Sk_P2Q9sG
def decompose_uncertainty(predictions, T, apply_Softmax=False):
    
    # Softmax usually already done in sample_weights
    if apply_Softmax == True:
        predictions = F.Softmax(predictions, dim=1)
        
    # Use biased variance because it the equivalent to calculations done by Shridhar
    epistemic_uncertainty = predictions.var(dim=1, unbiased=False)
    aleatoric_uncertainty = (predictions - predictions**2).mean(dim=1)

    return epistemic_uncertainty, aleatoric_uncertainty 

# Calculates beta to control the weight of complexity cost compared to the
# likelihood cost  
def get_beta(batch_idx, M, beta_type, epoch=None, num_epochs=None):
    # M = number of minibatches
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        # Formula: 
        #beta = 2 ** (M - (batch_idx + 1)) / ((2 ** M ) - 1)
        
        # Take log of exponent to prevent numerical overflow
        beta = np.exp(((M - (batch_idx + 1)) * np.log(2)) -  ((M - 1) * np.log(2)))
        
        # Prevent numerical underflow
        if beta < 1e-300:
            beta = 1e-300
            
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / M
    else:
        beta = 0
    return beta


