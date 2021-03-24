
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
import neptune
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from systems.Mixture_of_gaussians.ResNet import ResNet18Dec
from utils_.dataloader import ECGDataset
import json
import pickle
from datetime import datetime
from torch.utils.data import DataLoader


config = json.load(open('./ecgresnet_config.json', 'r'))

class Decoder(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    """
    def __init__(self,net,**args):
        """
        Args:
            beta: weight for kl term of the loss
            lr: learning rate for Adam, note: Adam is hardcoded
        """

        super(Decoder, self).__init__()

        self.lr = args['learning_rate']
        self.beta = args['beta']

        self.net = net      
        self.down_layer = nn.Sequential(nn.Linear(5000  * 8,400), nn.LeakyReLU(), nn.Linear(400, args['latent_dim']), nn.LeakyReLU() )
        #self.net = nn.Sequential(nn.Linear( args['latent_dim'], 400), nn.LeakyReLU(),nn.Linear( 400, 201 * 2 * 8) )

        self.show_figures = args['visualize'] == "True"

        filename = './systems/Mixture_of_gaussians/checkpoints/pretrained_decoder.sav'
        pickle.dump(self.net, open(filename, 'wb'))

        if self.show_figures:
            plt.ion()

            self.fig, self.axs = plt.subplots(8)
            x = range(0,5000)
            y = np.sin(x)
            self.line = [0] * 8

            for channel in range(0,8):
                self.line[channel], = self.axs[channel].plot(x, y, 'r-')
        
        
        print("Decoder initialized for pretraining")



    def forward(self, x):

        x = self.down_layer(torch.flatten(x, 1))
        x = F.softmax(self.net(x))
        return x

    def _run_step(self, x):

        x = self.down_layer(x)
        x = self.net(x)
        return x


    def step(self, batch, batch_idx):

        x  = batch['waveform'].float()

        #x = torch.rfft(x, signal_ndim =1, normalized = True)
        x_in = torch.flatten(x, 1)

        x_hat = self._run_step(x_in)
        x_hat = x_hat.reshape(x_hat.shape[0], 8, 5000)
        if self.show_figures and batch_idx%20 == 0:
            '''
            data_to_vis = x_hat[0].reshape(8, 201, 2)
            self.visualize_data_point(torch.irfft(data_to_vis, signal_ndim = 1))
            self.visualize_data_point(data_to_vis)
            '''
            self.visualize_data_point(x_hat[0])

        loss = F.mse_loss(x_hat, x)
        logs = {
            'loss' : loss
        }
        return loss, logs
        
    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        

        self.log_dict({f"train_{k}": v
                       for k, v in logs.items()},
                      on_step=True,
                      on_epoch=False)
        return loss

    def training_epoch_end(self, training_step_out):
        avg_train_loss = torch.mean(
            torch.Tensor([o['loss'] for o in training_step_out]))
        print('average epoch loss - train: ' + str(avg_train_loss))



        ### save the net to be transfered to the VAE
        filename = './systems/Mixture_of_gaussians/checkpoints/pretrained_decoder.sav'
        pickle.dump(self.net, open(filename, 'wb'))

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def validation_epoch_end(self, validation_step_out):
        pass
        '''
        avg_valid_loss = torch.mean(
            torch.Tensor([o for o in validation_step_out]))
        print('average epoch loss - valid: ' + str(avg_valid_loss))


        '''
        
       
    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, test_out):
        pass
        '''
        avg_test_acc = torch.mean(
            torch.Tensor([o['test_accuracy'] for o in test_out]))
        print('average test accuracy: ' + str(avg_test_acc))

        '''
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def save_model(self):
        pass


    def visualize_data_point(self, data_point, title = "Generated data example"):
        #data_point = torch.narrow(data_point, 1, 0,400)
        data_point = data_point.cpu().detach().numpy()
        if data_point.shape[0] > 1:
            for channelInd in range(data_point.shape[0]):
                channel_data = data_point[channelInd]
                self.axs[channelInd].set_ylim(np.min(channel_data), np.max(channel_data))
                self.line[channelInd].set_ydata(channel_data)
                self.axs[channelInd].set_title("Channel " + str(channelInd + 1))
        else:
            plt.plot(data_point[0])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

    ########################################
    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--beta", type=float, default=0.1)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

def train_model(model, data, modes, epoch_num,  args=None):

    pl.seed_everything(42)
    parser = ArgumentParser()
    script_args, _ = parser.parse_known_args(args)


    parser = model.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    ### general: set gpu to be used
    ##args.gpus = 1


    print("arguments parsed")
        
    trainer = pl.Trainer.from_argparse_args(args, track_grad_norm=2, max_epochs = epoch_num)

    print("trainer set")

    if "train" in modes:
        trainer.fit(model, data)
    if "test" in modes:
        trainer.test(model=model, datamodule=data)
    return data, model, trainer


def pretrain_decoder(data, epoch_num, from_pretrained = False, checkpoint_path_ = None, merged_dict = None):
    if from_pretrained:
        subnet = pickle.load(open(config['from_pretrained_decoder'], 'rb'))
    else:
        subnet = ResNet18Dec(**config)

    if checkpoint_path_:
        merged_dict['net'] = subnet
        model = Decoder.load_from_checkpoint(checkpoint_path_, **merged_dict)
    else:
        model = Decoder(subnet, **config)

    train_model(model,data, ['train'], epoch_num = epoch_num )
    return model.net

from torchvision import transforms
from utils_.transforms import ToTensor, Resample
from utils_.transforms import ApplyGain
transform = transforms.Compose([ToTensor(), ApplyGain(umc=False)])

dataset_params = json.load(open('./configs/UMCU-Triage.json', 'r'))
trainset = ECGDataset(path_labels_csv = dataset_params['train_labels_csv'],
                        waveform_dir = dataset_params['data_dir'],
                        OOD_classname = str(dataset_params['OOD_classname']),
                        transform = transform,
                        label_column = 'Label')

train_loader = DataLoader(trainset, batch_size=config['batch_size'], num_workers=8)
merged_dict = {**config, **dataset_params}

pretrain_decoder(train_loader, 30, checkpoint_path_= config['checkpoint_path_'], merged_dict=merged_dict)