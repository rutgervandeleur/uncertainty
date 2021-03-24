### This is inteded to be an easily modifiable VAE parent class for everyone who uses VAE-models
### Based on the pytorch lightning bolts VAE code
### neptune.AI support added (specifiy username and token in input var)

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
import neptune
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy.random import choice
import time
import pickle
import random
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import seaborn as sn
from utils_.tsne import tsne
import copy
import sys
sys.path.append('../..')
from systems.ecgresnet_uncertainty import ECGResNetUncertaintySystem
from systems.Mixture_of_gaussians.latent_probability_models import Mixture_of_Gaussians
from systems.Mixture_of_gaussians.ResNet import ResNet18Dec
from datetime import datetime

class VAE_with_Mixture_of_Gaussians(pl.LightningModule):
    """
    VAE with mixture of gaussians for classification and uncertainty estimation.
    """
    def __init__(self,**args ):
        
        super(VAE_with_Mixture_of_Gaussians, self).__init__()

        print("args is")
        print(args)

        self.lr = args['learning_rate']
        self.beta = args['beta']
        self.batch_size = args['batch_size']
        self.latent_dim = args['latent_dim']
        self.show_figures = args['visualize'] == "True"
        self.losses = args['losses']
        self.current_batch_idx  = 1

        args['include_classification'] = False

        self.encoder = ECGResNetUncertaintySystem(**args)
        if 'pretrained_decoder' in args:
            self.decoder = pickle.load(open(args['pretrained_decoder'], 'rb'))
            
            i = 0
            for param in self.decoder.parameters():
                if (i<5):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                i += 1

        else:
            self.decoder = ResNet18Dec(**args)
        self.latent_probability_model = Mixture_of_Gaussians(args['latent_dim'], args['encoder_output_dim'])
        #self.init_weights([self.encoder, self.decoder, self.latent_probability_model])
        if self.show_figures :
            self.figure   = self.initialize_figure(args['data_dir'])

        self.cm_all = torch.zeros(6,6)
        self.cm_certain = torch.zeros(6,6)
        self.cm_uncertain = torch.zeros(6,6)
        self.acc = []
        self.uncertain_prediction = []
        self.certain_prediction = []
        self.predictions_certain = []
        self.true_labels_certain = []
        self.predictions_all = []
        self.true_labels_all = []
        self.prediction_uncertainties = []
        self.train_acc =0 
        self.best_mF1 = 0
        self.cat_weights = [1,1,1,1,1,1]


    
        self.configure_optimizers()
        self.initialize_classifier(args['latent_dim'])


        ### init neptune for logging
        project_name = "balinthompot/VAE-test"
        neptune_token_path = "/workspace/ecgnet/api_key.txt"
        neptune_test_name = 'ecg-test-'
        neptune_token =  open(neptune_token_path, "r").read()
        neptune.init(project_qualified_name=project_name,
                    api_token=neptune_token)

        neptune.create_experiment(name=neptune_test_name +
                                datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                params=args)


        print("VAE initialized")

    def initialize_classifier(self, latent_dim):
        self.classification_layer = nn.Sequential(nn.Linear(latent_dim , 6))


    def initialize_figure(self, data_dir):
            plt.ion()

            fig, axs = plt.subplots(8, 4, figsize=(20,30))

            if data_dir == "/raw_data/umcu_median":
                input_width = 600
            else:
                
                input_width = 5000
       

            line = []
            
            for sub_fig in range(4):
                for channel in range(0,8):
                    x = range(0,input_width)
                    y = choice(range(-500, 500), input_width)
                    new_line, = axs[channel,sub_fig].plot(x, y, 'r-')
                    axs[channel,sub_fig].set_title("Channel " + str(channel + 1))
                    line.append(new_line)

            axs[0,0].annotate('Generated', xy=(0.5, 1), xytext=(0, 30),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

            axs[0,1].annotate('Reconstructed', xy=(0.5, 1), xytext=(0, 30),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

            axs[0,2].annotate('Original', xy=(0.5, 1), xytext=(0, 30),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
     
            axs[0,3].annotate('Difference', xy=(0.5, 1), xytext=(0, 30),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

            data = {
                'figure':fig,
                'axes' : axs,
                'lines':line
            }

            return data

    def init_weights(self, models):
        for m in models:
            for p in m.parameters():
                if len(p.data.shape) > 1:
                    torch.nn.init.xavier_normal(p.data)
                else:
                    torch.nn.init.normal(p.data)

    def forward(self, x):
        aux,x = self.encoder(x)
        latent_params = self.latent_probability_model.get_latent_params(x)
        p, q, z = self.sample(latent_params)
        return self.decoder(z)

    def _run_step(self, x):

        aux,x = self.encoder(x)
        #print(x)
        #print("----")
        #print(x.shape)
        #print("---")
        latent_params = self.latent_probability_model.get_latent_params(x)
        #print("++++++")
        #print(latent_params)
        #print("++++++")
        p, q, z = self.sample(latent_params)

        #print(y_hat)
        #exit()
        
        return z, p, q


    def sample(self, latent_params):
        p = self.latent_probability_model.p
        q = self.latent_probability_model.q(latent_params)
        z = self.latent_probability_model.get_z(q)

        return p, q, z

    
    def get_expert_features(self,batch):
        keys = list(batch.keys())
        m = torch.stack(([batch[label] for label in keys[4:]]), 0)
        y = m.T
        y = y.type(torch.FloatTensor).to(self.device)
        y -= exp_feat_means
        y /= exp_feat_stds

        return y

    def get_triage_mask(self, batch):
        
        triage_cats = [0,1,2,3,4,5]
        triage_labels = batch['label']
        batch_size = batch['waveform'].shape[0]

        ### replace cat 5 with cat 4, as in the paper

        triage_mask = torch.stack([ torch.stack([torch.full([self.latent_dim], self.cat_weights[triage_cat]) if triage_labels[ind] == triage_cat else torch.full([self.latent_dim], 0) for ind in range(batch_size)] , 0) for triage_cat in triage_cats], 0)
        
        return triage_mask.to('cuda')

    def step(self, batch, batch_idx):

        x = batch['waveform'].float()
        #y = self.get_expert_features(batch)

        self.batch = x
        self.batch_full = batch
    
        z, p, q = self._run_step(x)
           

        triage_mask = self.get_triage_mask(batch)
        kl, kl_all = self.latent_probability_model.KL(p,q,z, triage_mask)


        #self.beta = (torch.max(recon_loss) / kl).item()
        kl *= self.beta
        
        ## loss part to represent the expert features in the first dimension
        ##expert_feature_loss = F.mse_loss(z[:, :15], y) * 100

        if torch.isnan(kl).any():
            self.on_nan_loss(z,x,x_hat, "kl")
        y = batch['label']

        #kl_all = torch.reshape(kl_all, (kl_all.shape[1], kl_all.shape[0] * kl_all.shape[2]))
        ##y_hat = self.classification_layer(z)
        ##classification_loss = F.cross_entropy(y_hat , y)
        ##classification_loss = self.classification_loss(y_hat , y, True)
        neptune.log_metric('kl_loss', kl.item())

        loss =  kl ##+ classification_loss

        if 'reconstruction' in self.losses:
            recon = self.decoder(z)
            recon_loss = F.mse_loss(recon, x, reduction='sum')
            neptune.log_metric('recon_loss', recon_loss.item())

            self.recon = recon



            loss += recon_loss
        if self.show_figures and batch_idx%20 == 0:

            self.visualize_data_point(self.batch[0] , 2)

            if 'reconstruction' in self.losses:
                self.visualize_data_point(self.recon[0] , 1)
                self.visualize_data_point(self.generate_data()[0], 0)
                self.visualize_data_point(self.batch[0] - self.recon[0] , 3)

        logs = {

            "loss": loss,
        }
        
        neptune.log_metric('total_loss', loss)

        self.current_batch_idx += 1

        return loss, logs
        
    def on_nan_loss(self, z, x, x_hat, loss_type):
            print(loss_type + " is nan")
            print("max of z")
            print(torch.max(z))
            print("max of x")
            print(torch.max(x))
            print("max of x_hat")
            print(torch.max(x_hat))
            print("----")
            print("min of z")
            print(torch.min(z))
            print("min of x")
            print(torch.min(x))
            print("min of x_hat")
            print(torch.min(x_hat))
            print("z:")
            print(z)
            exit()

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)

        self.log_dict({f"train_{k}": v
                       for k, v in logs.items()},
                      on_step=True,
                      on_epoch=False)
        
        ##self.calc_data_probability(copy.deepcopy(batch))

        return loss

    def training_epoch_end(self, training_step_out):
        avg_train_loss = torch.mean(
            torch.Tensor([o['loss'] for o in training_step_out]))

        print('average epoch loss - train: ' + str(avg_train_loss))



    def validation_step(self, batch, batch_idx):
        ###self.calc_data_probability(batch)
        x = batch['waveform'].float()
        z, p, q = self._run_step(x)
        self.calc_data_probability(batch)
        return 0

    def validation_epoch_end(self, validation_step_out):
        self.ranked_evaluation()
        self.visualize_confusion_matrices()

        self.cm_all =  torch.zeros(6,6)
        self.cm_certain =  torch.zeros(6,6)
        self.cm_uncertain =  torch.zeros(6,6)
        self.acc = []
        self.uncertain_prediction = []
        self.certain_prediction = []
        self.predictions_certain = []
        self.true_labels_certain = []
        self.train_acc = 0
        self.current_batch_idx = 0
        self.prediction_uncertainties = []
        self.predictions_all = []
        self.true_labels_all = []

       
    def test_step(self, batch, batch_idx):
        x = batch['waveform'].float()
        z, p, q = self._run_step(x)

        logs = {
            "test_accuracy": 0,  ###TODO
        }
        self.calc_data_probability(batch)
        return logs

        ##TODO

    def test_epoch_end(self, test_out):
        self.ranked_evaluation()
        self.visualize_confusion_matrices()
        self.cm_all =  torch.zeros(6,6)
        self.cm_certain =  torch.zeros(6,6)
        self.cm_uncertain =  torch.zeros(6,6)
        self.acc = []
        self.uncertain_prediction = []
        self.certain_prediction = []
        self.predictions_certain = []
        self.true_labels_certain = []
        self.prediction_uncertainties = []
        self.predictions_all = []
        self.true_labels_all = []

        self.train_acc = 0
        filename = './VAE-base/checkpoints/pretrained/pretrained_encoder.sav'

        #pickle.dump(self.encoder, open(filename, 'wb'))


        '''
        avg_test_acc = torch.mean(
            torch.Tensor([o['test_accuracy'] for o in test_out]))
        print('average test accuracy: ' + str(avg_test_acc))

        '''

    def unfreeze_subnet(self,net):
        for param in net.parameters():
            param.requires_grad = True
    
    def freeze_subnet(self, net):
        for param in net.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        ### if this is a new net, we freeze the enc/dec for the first few iteration
        
        ##self.freeze_subnet(self.encoder)
        #self.freeze_subnet(self.decoder)
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    
    ### TODO ############################

    def generate_data(self,z_original = None, number_of_samples=1):
        if z_original == None:
            z = self.latent_probability_model.sample_prior(number_of_samples)
        else:
            mult = torch.ones(z_original[0].shape).to('cuda')
            mult[4] = -1
            z = z_original[0] * mult
            z = torch.unsqueeze(z, 0)

        return self.decoder(z)

    def visualize_data_point(self, data_point, figure_axis):
        data_point = data_point.cpu().detach().numpy()

        for channelInd in range(data_point.shape[0]):
            channel_data = data_point[channelInd]
            self.figure['axes'][channelInd, figure_axis].set_ylim(np.min(channel_data), np.max(channel_data))
            self.figure['lines'][channelInd + 8 *figure_axis].set_ydata(channel_data)
        self.figure['figure'].canvas.draw()
        self.figure['figure'].canvas.flush_events()

    def calc_confusion_matrix(self, true_labels, predicted_labels):
        cm = np.zeros((6,6))
        for ind in range(true_labels.shape[0]):
            cm[true_labels[ind]][predicted_labels[ind]] += 1
        return cm

    def calibration_evaluation(self):

        self.prediction_uncertainties = np.asarray(self.prediction_uncertainties)
        self.predictions_all = np.asarray(self.predictions_all)
        self.true_labels_all = np.asarray(self.true_labels_all)

        prediction_order_by_uncertainty = self.prediction_uncertainties.argsort()
        ordered_predictions = self.predictions_all[prediction_order_by_uncertainty]
        ordered_labels = self.true_labels_all[prediction_order_by_uncertainty]
        
        bin_size = 1
        cutoff_point = 1
        accuracies = []
        fraction = []
        while cutoff_point < np.size(ordered_predictions):
            a = ordered_predictions[:cutoff_point] == ordered_labels[:cutoff_point]
            accuracies.append(np.sum(a) / len(a))
            fraction.append(cutoff_point / np.size(ordered_predictions))
            cutoff_point += bin_size
        fig = plt.figure(5)
        plt.xlabel("Fraction of included data points")
        plt.ylabel("Accuracy")

        plt.plot(fraction, accuracies)
        plt.show()

    def ranked_evaluation(self):

        self.prediction_uncertainties = np.asarray(self.prediction_uncertainties)
        self.predictions_all = np.asarray(self.predictions_all)
        self.true_labels_all = np.asarray(self.true_labels_all)

        prediction_order_by_uncertainty = self.prediction_uncertainties.argsort()

        ordered_predictions = self.predictions_all[prediction_order_by_uncertainty]
        ordered_labels = self.true_labels_all[prediction_order_by_uncertainty]

        twentyfive_p = False
        fifty_p = False
        seventyfive_p = False

        bin_size = 10
        cutoff_point = bin_size
        accuracies = []
        binned_acc = []
        fraction = []
        while cutoff_point < np.size(ordered_predictions):
            a = ordered_predictions[:cutoff_point] == ordered_labels[:cutoff_point]
            a_binned= ordered_predictions[cutoff_point - bin_size:cutoff_point] == ordered_labels[cutoff_point-bin_size:cutoff_point]
            ac = np.sum(a) / len(a)
            ac_binned = np.sum(a_binned) / len(a_binned)
            accuracies.append(ac)
            binned_acc.append(ac_binned)
            f = cutoff_point / np.size(ordered_predictions)
            fraction.append(f)
            if f >= 0.25 and not twentyfive_p:
                print("acc at 75 percent removed: " + str(ac))
                twentyfive_p = True
            if f >= 0.5 and not fifty_p:
                print("acc at 50 percent removed: " + str(ac))
                fifty_p = True

            if f >= 0.75 and not seventyfive_p:
                print("acc at 25 percent removed: " + str(ac))
                seventyfive_p = True
                
            cutoff_point += bin_size

        fig, axs = plt.subplots(1, 2, figsize=(20,30))
        axs[0].plot(fraction, accuracies)
        axs[ 0].set_title('Cumulative')
        axs[0].set_xlabel("Fraction of included data points")
        axs[0].set_ylabel("Accuracy")
      
        axs[1].plot(fraction, binned_acc)
        axs[1].set_title('Binned')
        axs[1].set_xlabel("Bin")
        axs[1].set_ylabel("Accuracy")
        plt.show()

        plt.savefig("ranked_eval.png")



    def visualize_confusion_matrices(self):
        
        triage_cat_names = ['Normal', 'Not acute', 'Subacute', 'Acute-arrythmia', 'Acute-ischaemia', 'Normal-pacemaker']

        self.acc = torch.cat(self.acc)
        self.uncertain_prediction = torch.cat(self.uncertain_prediction)
        self.certain_prediction = torch.cat(self.certain_prediction)
        print("train accuracy: ")
        print(self.train_acc / self.current_batch_idx)
        print("prediction accuracy: ")
        print(torch.sum(self.acc) / self.acc.shape[0])
        print("accuracy on certain prediction - " + str(torch.sum(self.certain_prediction)))
        print(torch.sum(self.acc * self.certain_prediction) / torch.sum(self.certain_prediction))
        print("accuracy on uncertain prediction - " + str(torch.sum(self.uncertain_prediction)))
        print(torch.sum(self.acc * self.uncertain_prediction) / torch.sum(self.uncertain_prediction))
        

        cm_all_df = pd.DataFrame(self.cm_all, index = triage_cat_names,
                  columns = triage_cat_names)
        cm_all_df = cm_all_df[cm_all_df.columns].astype(float)

        cm_certain_df = pd.DataFrame(self.cm_certain, index = triage_cat_names,
                  columns = triage_cat_names)
        cm_certain_df = cm_certain_df[cm_certain_df.columns].astype(float)

        cm_uncertain_df = pd.DataFrame(self.cm_uncertain, index = triage_cat_names,
                  columns = triage_cat_names)
        cm_uncertain_df = cm_uncertain_df[cm_uncertain_df.columns].astype(float)


        fig, ax = plt.subplots(1,3,  figsize=(20,30))
        fig.suptitle('Confusion matrices')

        ax[0].set_title("All predictions (" + str(self.acc.shape[0]) + ")")
        ax[1].set_title("Certain predictions (" + str(torch.sum(self.certain_prediction).item()) + ")")
        ax[2].set_title("Uncertain predictions (" + str( torch.sum(self.uncertain_prediction).item()) + ")")


        sn.heatmap(cm_all_df, annot=True, ax = ax[0])
        sn.heatmap(cm_certain_df, annot=True, ax = ax[1])
        sn.heatmap(cm_uncertain_df, annot=True, ax = ax[2])


        print("Macro F1 score")
        mF1 = f1_score(self.true_labels_certain, self.predictions_certain, average='macro')
        print(mF1)
        F1_class = f1_score(self.true_labels_certain, self.predictions_certain, average=None)

        print("F1 score per class")
        print(F1_class)
        '''
        for class_ind in range(len(F1_class)):
            self.cat_weights[class_ind] = 1-F1_class[class_ind]
        '''
        if self.best_mF1 < mF1:
            self.best_mF1 = mF1
            #print("saving model")
            #filename = '/workspace/uncertainty/project/systems/Mixture_of_gaussians/checkpoints/pretrained_full.sav'
            #pickle.dump(self, open(filename, 'wb'))
        print("Weighted F1 score")
        print(f1_score(self.true_labels_certain, self.predictions_certain, average='weighted'))

        print("showing")
        plt.savefig("test_res.png")
        plt.show()


    def calc_data_probability(self, batch):
        inp = batch['waveform']
        ###inp += ((torch.rand(inp.shape) - 0.5) * 1000).to('cuda')  
 
        aux,x = self.encoder(inp) 
    
        triage_labels = batch['label']
        
        latent_params = self.latent_probability_model.get_latent_params(x)
        p = self.latent_probability_model.p
        q = self.latent_probability_model.q(latent_params)

        
        mc_sample_num = 1024
        samples = q.sample([mc_sample_num])
        triage_cats = [0,1,2,3,4, 5]
        #class_prior_probs = torch.Tensor([0.5, 0.5, 0.2, 0.1])
        triage_cat_names = ['Normal', 'Not acute', 'Subacute', 'Acute']

        probs_prior = torch.stack([p[triage_cat].log_prob(samples) for triage_cat in triage_cats],0)
        latent_weights = q.log_prob(samples) 
        probs_weighted = probs_prior + latent_weights

        
        probs = torch.sum(probs_prior + latent_weights ,(1,3))
        full_prob_mass =  torch.logsumexp((probs), 0) 
        full_prob_mass /= torch.min(full_prob_mass)
        #probs = self.classification_layer(x)
        probs_std = torch.std(probs_prior + latent_weights ,(1,3))
        #probs_entropy = - torch.sum(torch.exp(probs_weighted) * probs_weighted ,(1,3))
        #print(probs_entropy)




        '''
        cat_id = 1
        
        idx_of_cat = triage_labels == cat_id
        print("cat probs:")
        print(torch.mean(probs[cat_id][idx_of_cat]))
        print(torch.mean(probs[cat_id][torch.logical_not(idx_of_cat)]))

        print("cat probs std:")
        print(torch.mean(probs_std[cat_id][idx_of_cat]))
        print(torch.mean(probs_std[cat_id][torch.logical_not(idx_of_cat)]))
        #exit()
        '''
        '''
        normal_prob_thresh = -1000000000
        not_acute_prob_thresh = -90000
        subacute_prob_thresh = -120000
        arrythmia_thresh = -200000
        isch_thresh = -200000
        pacemaker_thresh = -100000

        category_thresholds = [normal_prob_thresh, not_acute_prob_thresh, subacute_prob_thresh, arrythmia_thresh, isch_thresh, pacemaker_thresh]
        category_order = [5,4,3,2,1,0]

        max_probs = []
        for index in range(probs.shape[1]):
            for i in category_order:
                if probs[i][index] > category_thresholds[i]:
                    max_probs.append(i)
                    break

        max_probs = torch.LongTensor(max_probs).to('cuda')
        '''
        
        ##probs = probs * torch.Tensor([[0.5], [0.4], [0.3], [0.2]]).to('cuda')
        max_prob_vals,max_probs = torch.max(probs, 0)


        ##max_probs = torch.min(probs_std, 0)[1] + 1
        ##prob_vals = torch.max(probs, 0)[0]

        ##std_of_predicted = torch.stack([probs_std[triage_labels[instance] - 1][instance]  for instance in range(triage_labels.shape[1])])
        ##std_of_predicted = torch.stack([probs_std[triage_labels[instance] - 1][instance]  for instance in range(triage_labels.shape[1])])
        std_of_predicted = torch.stack([probs_std[max_probs[ind]][ind] for ind in range(len(max_probs))])
        std_of_predicted /= torch.max(std_of_predicted)
        #print(full_prob_mass)
        top2_std = torch.topk(probs_std, 2,0, largest=False)[0]
        top2_diff = top2_std[1] - top2_std[0]
        top2_prob = torch.topk(probs, 2,0, largest=False)[0]
        top2_diff_prob = top2_prob[1] - top2_prob[0]
        top2_diff_prob /= torch.max(top2_diff_prob)
        top2_diff_prob = 1- top2_diff_prob
        #print(top2_diff_prob)

        #std_of_predicted = (full_prob_mass * std_of_predicted) / (full_prob_mass + std_of_predicted)
        #print(full_prob_mass)
        #print(std_of_predicted)
        #print(top2_diff_prob)
        #print(full_prob_mass * std_of_predicted)
        std_of_predicted += full_prob_mass
        #std_of_predicted *= top2_diff_prob


        #print(std_of_predicted)


        
        ##std_of_predicted = torch.squeeze(std_of_predicted)

        ##std_of_predicted = torch.std(probs_std ,(0))
        
        acc = max_probs == triage_labels
        '''
        fig, axs = plt.subplots(3, 2, figsize=(20,30))
        axs[0, 0].scatter(probs[0].cpu(), probs_std[0].cpu(), c=(triage_labels == 0).cpu(), alpha=0.5)
        axs[0, 0].set_title('Normal')

        axs[0, 1].scatter(probs[1].cpu(), probs_std[1].cpu(), c=(triage_labels == 1).cpu(), alpha=0.5)
        axs[0, 1].set_title('Not acute')

        axs[1, 0].scatter(probs[2].cpu(), probs_std[2].cpu(), c=(triage_labels == 2).cpu(), alpha=0.5)
        axs[1, 0].set_title('Subacute')

        axs[1, 1].scatter(probs[3].cpu(), probs_std[3].cpu(), c=(triage_labels == 3).cpu(), alpha=0.5)
        axs[1, 1].set_title('Arrythmia')

        axs[2, 0].scatter(probs[4].cpu(), probs_std[4].cpu(), c=(triage_labels == 4).cpu(), alpha=0.5)
        axs[2, 0].set_title('Isch')

        axs[2, 1].scatter(probs[5].cpu(), probs_std[5].cpu(), c=(triage_labels == 5).cpu(), alpha=0.5)
        axs[2, 1].set_title('Pacemaker')

        plt.show()
        '''
        self.acc.append(acc)

        std_thresh = [1.6,1.6,1.6,1.6,1.6,1.6]
        #low_std = [std_of_predicted[i] < std_thresh[max_probs[i]] for i in range(max_probs.shape[0])]
        #low_std = torch.BoolTensor(low_std).to('cuda')
        low_std = (std_of_predicted < 1.8)
        #stds_apart = top2_diff > 0.7
        #entropy_of_predicted = torch.stack([probs_entropy[max_probs[ind]][ind] for ind in range(len(max_probs))])
        #print(entropy_of_predicted)
        #low_entropy = entropy_of_predicted < 2040
        certain_prediction = low_std #* stds_apart 

        '''
        tsne(p, samples[:,0,:], mc_sample_num, self.latent_dim )
        tsne(p, samples[:,1,:], mc_sample_num, self.latent_dim )
        tsne(p, samples[:,2,:], mc_sample_num, self.latent_dim )
        tsne(p, samples[:,3,:], mc_sample_num, self.latent_dim )
        '''
        uncertain_prediction = certain_prediction == False

        self.certain_prediction.append(certain_prediction)
        self.uncertain_prediction.append(uncertain_prediction)


        certainly_predicted_labes = max_probs[certain_prediction]
        uncertainly_predicted_labes = max_probs[uncertain_prediction]

        certainly_predicted_labes_true = triage_labels[certain_prediction]
        uncertainly_predicted_labes_true = triage_labels[uncertain_prediction]

        ### store to calculate F1 scores on certain preds, and ranked eval on all
        
        self.predictions_certain.extend(certainly_predicted_labes.cpu().detach().numpy())
        self.true_labels_certain.extend(certainly_predicted_labes_true.cpu().detach().numpy())
        self.predictions_all.extend(max_probs.cpu().detach().numpy())
        self.true_labels_all.extend(triage_labels.cpu().detach().numpy())
        self.prediction_uncertainties.extend(std_of_predicted.cpu().detach().numpy())
        
        ### confusion matrices
        cm_all = self.calc_confusion_matrix(triage_labels.cpu(), max_probs.cpu())
        self.cm_all += cm_all

        cm_certain = self.calc_confusion_matrix(certainly_predicted_labes_true.cpu(), certainly_predicted_labes.cpu())
        self.cm_certain += cm_certain

        cm_uncertain = self.calc_confusion_matrix(uncertainly_predicted_labes_true.cpu(), uncertainly_predicted_labes.cpu())
        self.cm_uncertain += cm_uncertain




  

    def save_model(self):
        pass

    ########################################
    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='VAE_mixture')
        parser.add_argument('--ensembling_method', type=bool, default=False)


        return parser
