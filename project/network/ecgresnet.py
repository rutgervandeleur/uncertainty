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

# import configuration
# ex = configuration.ex

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
        #return torch.cat((x, age.view(-1,1), gender.view(-1,1)), 1)

class BasicBlock(nn.Module):
    """
    This class implements a residual block.
    """
    def __init__(self, in_channels, out_channels, stride, dropout, dilation, num_branches):
        """
        Initializes BasicBlock object. 

        Args:
          in_channels: number of input channels
          out_channels: number of output channels
          stride: stride of the convolution
          dropout: probability of an argument to get zeroed in the dropout layer
        """
        super(BasicBlock, self).__init__()
        kernel_size = 5
        
        self.branch0 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels, in_channels // num_branches, kernel_size = 1, 
                      padding = 0, stride = 1,  bias = False),

            nn.BatchNorm1d(in_channels // num_branches),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels // num_branches, out_channels // num_branches, kernel_size = kernel_size, 
                      padding = (kernel_size - 1) // 2, stride = stride, bias = False)
        )
        
        self.branch1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels, in_channels // num_branches, kernel_size = 1, 
                      padding = 0, stride = 1, dilation = 1, bias = False),

            nn.BatchNorm1d(in_channels // num_branches),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels // num_branches, out_channels // num_branches, kernel_size = kernel_size, 
                      padding = ((kernel_size - 1) * dilation) // 2, stride = stride, 
                      dilation = dilation, bias = False)     
        )
        
        if in_channels == out_channels and stride == 1:
            self.shortcut = lambda x: x
            
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size = 1, 
                                      padding = 0, stride = stride, bias=False)
            

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

class ECGResNet(nn.Module):
    """
    This class implements the ECG-ResNet in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ResNet object can perform forward.
    """
    def __init__(self, in_length, in_channels, n_grps, N, num_classes, dropout, first_width, 
                 stride, dilation):

        # THESE ARGUMENTS ARE NOW UNDEFINED #######################
        # train_params, load_params, epistemic_method, aleatoric_method, device, train=False
        ###################################################################################

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
        # self.num_classes = num_classes
        # self.train_params = train_params
        # self.load_params = load_params
        # self.epistemic_method = epistemic_method 
        # self.aleatoric_method = aleatoric_method
        # self.device = device
        
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
       
        # Initialize loggers [VISDOM ONLY]
        # self.init_loggers()
        
        # ALL HANDLED BY PYTORCH LIGHTNING
        # if train == False:
        #     # Load checkpoint from file
        #         model_path = load_params['model_weights_directory'] + str(load_params['model_run']) + "/ecgnet_{}_{}_{}_epoch{}_model{}.pt".format(
        #             epistemic_method,
        #             aleatoric_method,
        #             train_params['experiment_name'],
        #             load_params['epoch'],
        #             load_params['model_idx'])

        #         checkpoint = torch.load(model_path, map_location = self.device)
        #         self.load_state_dict(checkpoint['model_state_dict'])
        #         self.eval()
        #         print('Model {} initialized'.format(load_params['model_idx']))
        
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
                module.train()
        
    # Takes n Monte Carlo samples by 
    def mc_sample(self, data, n_samples):
        samples = torch.empty((data.shape[0], n_samples, self.num_classes))
        
        for i in range(n_samples):
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
    
    
    
    def run_epoch(self, dataloader, loss, epoch_idx):
        print("Training model for epoch {}/{}".format(epoch_idx+1, self.train_params['epochs']))

        # Put in model in training mode
        self.train()

        # Loop over dataset
        for batch_idx, sample in enumerate(dataloader):

            # Get data
            data, target = sample['waveform'].to(self.device), sample['label'].to(self.device)

            # Perform step
            self.optimizer.zero_grad()
            output1, output2 = self(data)
            train_loss1 = loss(output1, target)
            train_loss2 = loss(output2, target)
            total_train_loss = (0.3 * train_loss1) + train_loss2
            total_train_loss.backward()
            self.optimizer.step()

            # Log metrics 
            self.log_metrics(epoch_idx, batch_idx, self.train_params['batch_size'],
                             len(dataloader), len(dataloader.dataset),
                             total_train_loss.item(), output2.detach(), target.detach())
            
            
            
    def eval_epoch(self, dataloader, loss, epoch_idx, ev):
        print("Testing model for epoch {}/{}".format(epoch_idx+1, self.train_params['epochs']))
        self.eval()

        # Setup cumulative tensors
        output_cum  = torch.empty((0, self.num_classes))
        target_cum  = torch.empty((0)).type(torch.long)

        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                
                # Get data
                data, target = sample['waveform'].to(self.device), sample['label'].to(self.device)

                # Perform step
                output1, output2 = self(data)
                total_test_loss = loss(output2, target)

                output_cum = torch.cat((output_cum, output2.data.detach().cpu()), 0)
                target_cum = torch.cat((target_cum, target.data.detach().cpu()), 0) 

                # Log metrics 
                self.meter_loss.add(total_test_loss.item())
                self.meter_accuracy.add(output2.detach(), target.detach())

            print('\nValidation loss: {:.6f}\t Validation accuracy: {:.6f}\n'.format(self.meter_loss.value()[0], self.meter_accuracy.value()[0]))
            # y = label_binarize(target_cum.numpy(), classes = [0,1,2,3,4,5])

            ex.log_scalar('testing.loss', self.meter_loss.value()[0], epoch_idx + 1)
            ex.log_scalar('testing.accuracy', self.meter_accuracy.value()[0], epoch_idx + 1)
            
            if self.train_params['OOD_classname'] == 'none':
                y = label_binarize(target_cum.numpy(), classes = np.arange(self.num_classes))
                auc = roc_auc_score(y, F.softmax(output_cum, dim=1).numpy(), average = None)
                
                for j, value in enumerate(auc):
                    ex.log_scalar("auc_{}".format(str(j)), value * 100, epoch_idx + 1)

            self.meter_accuracy.reset()
            self.meter_loss.reset()

    def predict_dataset(self, dataloader, loss, ev, expert_test_set, num_test_classes, split_expert_test_classes):
        
        predictions = torch.empty((0, num_test_classes))
        predictions_conf = torch.empty((0, num_test_classes))
        
        nll = torch.empty(0)
        mae = torch.empty(0)
        brier_score = torch.empty(0)
        ids = torch.empty(0).type(torch.LongTensor)
        labels = torch.empty(0).type(torch.LongTensor)
        
        predicted_labels = torch.empty(0).type(torch.LongTensor)
        predicted_labels_confidence = torch.empty(0).type(torch.FloatTensor)
        correct_predictions = torch.empty(0).type(torch.BoolTensor)
        
#        if (self.experiment_name != 'in'):
#            OOD_class_labels = torch.empty(0).type(torch.FloatTensor)
        softmax = nn.Softmax(dim=1)

        # Turn on dropout at so we have variation
        self.eval()
        self.enable_dropout()
        
        with torch.no_grad():
            for sample in dataloader:
                data = sample['waveform'].to(self.device)
                targets = sample['label'].to(self.device)
                
                # Make prediction
                output1, output2 = self(data)
                
                # Convert to expert UMCTriage expert test set
                if expert_test_set == True:
                    predicted_labels_no_sm, output2 = convert_predictions_to_expert_categories(output2, split_expert_test_classes)
                
                prediction = softmax(output2)
                    
                # Gather results
                predictions = torch.cat((predictions, prediction.data.cpu()), 0)
                
                nll = torch.cat((nll, ev.calc_negative_log_likelihood(output2.data.cpu(), targets.data.cpu(), per_batch=False).cpu()), 0)
                
                mae = torch.cat((mae, ev.calc_mean_absolute_error(prediction.data.cpu(), targets.data.cpu()).cpu()), 0)
                brier_score = torch.cat((brier_score, ev.calc_brier_score(prediction.data.cpu(), targets.data.cpu(), per_batch=False).cpu()), 0)
                ids = torch.cat((ids, sample['id']), 0)
                labels = torch.cat((labels, targets.cpu()), 0)
                predicted_labels = torch.cat((predicted_labels, prediction.argmax(dim=1).cpu()), 0)
                
                correct_predictions = torch.cat((correct_predictions, torch.eq(prediction.argmax(dim=1).data.cpu(), targets.data.cpu()).cpu()), 0)
                predicted_labels_confidence = torch.cat((predicted_labels_confidence, torch.gather(prediction, 1, prediction.argmax(dim=1).unsqueeze_(1))[:, 0].cpu()), 0)
        
        if self.train_params['OOD_classname'] == 'none':
                y = label_binarize(labels.numpy(), classes = np.arange(num_test_classes))
                auc = roc_auc_score(y, predictions.numpy(), average = None)
                
                for j, value in enumerate(auc):
                    ex.log_scalar("testing.auc_{}".format(str(j)), value * 100)
                    
        # Combine results into single dataframe 
        comb = pd.concat([
            pd.DataFrame(ids.numpy(), columns= ['ID']),  
            
            pd.DataFrame(predictions.numpy(), columns= ['mean_{}'.format(i) for i in range(num_test_classes)]), 
            pd.DataFrame(labels.numpy(), columns= ['labels']), 
            pd.DataFrame(predicted_labels.numpy(), columns= ['predicted_labels']),
            pd.DataFrame(predicted_labels_confidence.numpy(), columns= ['predicted_labels_confidence']), 
            pd.DataFrame(nll.numpy(), columns= ['cross_entropy_loss']), 
            pd.DataFrame(mae.numpy(), columns= ['mean_absolute_error']), 
            pd.DataFrame(brier_score.numpy(), columns= ['brier_score']), 
            pd.DataFrame(correct_predictions.numpy(), columns= ['correct_prediction']),
            pd.DataFrame(predicted_labels_confidence.numpy(), columns= ['epistemic_uncertainty']), 
        ], axis=1)
        
        print(comb)
        return comb

    def log_metrics(self, epoch_idx, batch_idx, batch_size, n_batches, n_samples, loss, output, target):

        self.meter_loss.add(loss)
        self.meter_accuracy.add(output, target)

        if batch_idx % self.train_params['log_interval'] == 0:
            observed_samples = (epoch_idx * n_samples) + ((batch_idx + 1) * batch_size)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                epoch_idx + 1,
                (batch_idx + 1) * batch_size,
                n_samples,
                100. * ((batch_idx + 1) / n_batches),
                self.meter_loss.value()[0],
                self.meter_accuracy.value()[0]))

            ex.log_scalar('training.loss', self.meter_loss.value()[0], observed_samples)
            ex.log_scalar('training.accuracy', self.meter_accuracy.value()[0], observed_samples)

            self.meter_accuracy.reset()
            self.meter_loss.reset()

    def save(self, epoch, loss, model_params, epistemic_method, aleatoric_method):
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'model_params': model_params,
                'train_params': self.train_params
            }, self.train_params['save_location'] +
            str(self.train_params['now']) +
            "/ecgnet_{}_{}_{}_epoch{}_model1.pt".format(epistemic_method,
                                                        aleatoric_method,
                                                        self.train_params['experiment_name'], epoch + 1))
        print("Model :" + str(self.train_params['now']) +
              "/ecgnet_{}_{}_{}_epoch{}model1.pt".format(epistemic_method,
                                                         aleatoric_method,
                                                         self.train_params['experiment_name'], epoch + 1))



    # Initializes loggers 
    def init_loggers(self, model_params, train_params, visdom_log):
        num_labels = model_params['num_classes']
        now = train_params['now']
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        self.confusion_meter = tnt.meter.ConfusionMeter(num_labels, normalized=True)

        # Initialize visdom loggers
        if visdom_log == True:
            self.train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss', 
                                                          'xlabel': 'Epoch',
                                                          'ylabel': 'Crossentropy loss'},
                                            env = '{}-{}_{}_{}_{}_{}'.format(architecture,
                                                                                       str(self.n_samples), self.epistemic_method, self.aleatoric_method, self.experiment_name, now))
            self.train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy',
                                                           'xlabel': 'Epoch',
                                                           'ylabel': 'Accuracy'},
                                            env = '{}-{}_{}_{}_{}_{}'.format(architecture,
                                                                                       str(self.n_samples), self.epistemic_method, self.aleatoric_method, self.experiment_name, now))
            
            self.test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss', 
                                                          'xlabel': 'Epoch',
                                                          'ylabel': 'Crossentropy loss'},
                                            env = '{}-{}_{}_{}_{}_{}'.format(architecture,
                                                                                       str(self.n_samples), self.epistemic_method, self.aleatoric_method, self.experiment_name, now))
            self.test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy', 
                                                          'xlabel': 'Epoch',
                                                          'ylabel': 'Accuracy'},
                                            env = '{}-{}_{}_{}_{}_{}'.format(architecture,
                                                                                       str(self.n_samples), self.epistemic_method, self.aleatoric_method, self.experiment_name, now))
            self.test_confusion_logger = VisdomLogger('heatmap', opts={'title': 'Ensemble: Test Confusion matrix', 
                                                          'xlabel': 'Predicted',
                                                          'ylabel': 'Ground Truth',
                                                          'columnnames': list(range(num_labels)),
                                                          'rownames': list(range(num_labels))},
                                            env = '{}-{}_{}_{}_{}_{}'.format(architecture,
                                                                                       str(self.n_samples), self.epistemic_method, self.aleatoric_method, self.experiment_name, now))
            self.uncertainty_logger = VisdomPlotLogger('line',
                                                       opts={'title':'Uncertainty measures',
                                                             'xlabel':'Epoch',
                                                             'ylabel':'Measure'},
                                                       env = '{}-{}_{}_{}_{}_{}'.format(architecture,
                                                                                       str(self.n_samples), self.epistemic_method, self.aleatoric_method, self.experiment_name, now))
            self.test_roc_auc_logger = VisdomPlotLogger('line',
                                                        opts={'title':'Ensemble ROC AUC',
                                                              'xlabel':'Epoch',
                                                              'ylabel':'Accuracy'},
                                                        env = '{}-{}_{}_{}_{}_{}'.format(architecture,
                                                                                       str(self.n_samples), self.epistemic_method, self.aleatoric_method, self.experiment_name, now))

            self.train_confusion_logger = VisdomLogger('heatmap', opts={'title': 'Model {}: Train Confusion matrix'.format(i+1), 
                                                      'xlabel': 'Predicted',
                                                      'ylabel': 'Ground Truth',
                                                      'columnnames': list(range(num_labels)),
                                                      'rownames': list(range(num_labels))},
                                        env ='{}-{}_{}_{}'.format(architecture, self.epistemic_method, self.experiment_name, now))
            self.test_confusion_logger = VisdomLogger('heatmap', opts={'title': 'Model {}: Test Confusion matrix'.format(i+1), 
                                                      'xlabel': 'Predicted',
                                                      'ylabel': 'Ground Truth',
                                                      'columnnames': list(range(num_labels)),
                                                      'rownames': list(range(num_labels))},
                                       env = '{}-{}_{}_{}'.format(architecture, self.epistemic_method, self.experiment_name, now))

