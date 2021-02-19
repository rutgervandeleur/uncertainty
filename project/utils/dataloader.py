################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Rutger van de Leur
# Date Created: 2018-12-10
################################################################################

from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.helpers import *
import scipy
import scipy.io
from scipy import signal
import sys
from icecream import ic
import random
from scipy import misc
import matplotlib.pyplot as plt

################################################################################

class ECGDataset(Dataset):
    def __init__(self, path_labels_csv, waveform_dir, OOD_classname, label_column, transform=None, assert_shape=True, get_age=True, get_gender=True):
        """
        Args:
            path_labels_csv (string): Path to the csv file with PseudoID, TestID and Label.
            waveform_dir (string): Directory with all the waveforms.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if path_labels_csv is not None:
            self.umc = True
            self.path_labels = pd.read_csv(path_labels_csv)
        else:
            self.umc = False
            self.path_labels = sorted(os.listdir(waveform_dir))
        
        self.waveform_dir = waveform_dir
        self.transform = transform
        self.OOD_classname = OOD_classname
        self.label_column  = label_column
        self.get_age = get_age
        self.get_gender = get_gender

    def __len__(self):
        return len(self.path_labels)

    def __getitem__(self, idx, assert_shape=True):
        if self.umc:
            waveform = np.load(os.path.join(self.waveform_dir, generate_path(self.path_labels['PseudoID'].iloc[idx]), str(self.path_labels['TestID'].iloc[idx])) + '.npy')
            if assert_shape==True:
                if assert_shape_match(waveform, (8, 5000)):
                    label = self.path_labels[self.label_column].iloc[idx]
                    age = self.path_labels['Age'].iloc[idx] if self.get_age else 0
                    gender = self.path_labels['Gender'].iloc[idx] if self.get_gender else 0
                    id = self.path_labels['TestID'].iloc[idx]
                else:
                    return None
            else:
                label = self.path_labels[self.label_column].iloc[idx]
                age = self.path_labels['Age'].iloc[idx] if self.get_age else 0
                gender = self.path_labels['Gender'].iloc[idx] if self.get_gender else 0
                id = self.path_labels['TestID'].iloc[idx]
                
        else:
            waveform = np.loadtxt(os.path.join(self.waveform_dir, self.path_labels[idx]), delimiter = ',').transpose()
            label = 0
            age = 0
            gender = 0
            id = self.path_labels[idx]

        if self.OOD_classname != 'none':                     
            sample = {'waveform': waveform, 'label': label, 'age': age, 'gender': gender, 'id': id, 'OOD_class': self.OOD_classname}
        else:
            sample = {'waveform': waveform, 'label': label, 'age': age, 'gender': gender, 'id': id}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

# class ToTensor(object):
#     """
#     Convert ndarrays in sample to Tensors.
#     """

#     def __call__(self, sample):
#         # waveform = sample['waveform'][0:8,]
#         waveform = sample['waveform']
#         sample['waveform'] = torch.from_numpy(waveform).type(torch.FloatTensor)
#         sample['age'] = torch.from_numpy(np.array(sample['age'])).type(torch.FloatTensor)
#         sample['gender'] = torch.from_numpy(np.array(sample['gender'])).type(torch.FloatTensor)
#         return sample
    
# class ApplyGain(object):
#     """
#     Normalize ECG signal by multiplying by specified gain and converting to millivolts
#     """
#     def __init__(self, umc = True):
#         self.umc = umc
        
#     def __call__(self, sample, umc = True):
#         if self.umc:
#             waveform = sample['waveform'] * 0.001 * 4.88
#         else:
#             waveform = sample['waveform'] * 0.001
#         sample['waveform'] = waveform
        
#         return sample
    
# class SelectMiddleN(object):
#     """
#     Select middle n timepoints of ECG signal
#     """
#     def __init__(self, n):
#         self.n = n
        
#     def __call__(self, sample):
#         waveform = sample['waveform']
#         start = (waveform.shape[1] - self.n) // 2
#         stop = start + self.n
#         sample['waveform'] = waveform[:,start:stop]
        
#         return sample
    
# class SwitchLeads(object):
#     """
#     Switch leads between specified indices.
#     Leads are in following order: [I, II, V1, V2, V3, V4, V5, V6]
#     """
    
#     def __init__(self, lead_idx=[0, 1]):
#         self.lead_idx = lead_idx
    
#     def __call__(self, sample):
#         # Extract leads at indices
#         lead_0 = torch.clone(sample['waveform'][self.lead_idx[0]]) # Needs to be cloned, otherwise lead_0 will be overwritten by first switch step
        
#         # Switch leads
#         sample['waveform'][self.lead_idx[0]] = sample['waveform'][self.lead_idx[1]]
#         sample['waveform'][self.lead_idx[1]] = lead_0

#         return sample

# def collate_fn_skip_None(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch)

# def assert_shape_match(data, shape):
#     if data.shape == shape:
#         return True
#     else:
#         return False

class CPSC2018Dataset(Dataset):
    '''China Physiological Signal Challenge 2018 dataset'''
    def __init__(self, path_labels_csv, waveform_dir, OOD_classname,
                 transform=None, assert_shape=True, max_sample_length=5000):
        '''
        Args:
        path_labels_csv (string): Path to csv file containing names of
        datapoints with associated labels
        waveform_dir (string): Directory with all the datapoints.
        OOD_classname (string): Name of the out-of-distribution to be excluded
        during training
        transform (callable, optional): Optional transform to be applied
        on a sample.
        assert_shape (bool): Whether to assert the correct shape of waveform
        '''
        self.path_labels = pd.read_csv(path_labels_csv)
        self.waveform_dir = waveform_dir
        self.transform = transform
        self.OOD_classname = OOD_classname
        self.classes = self.get_classes()
        self.num_classes = len(self.classes)
        self.max_sample_length = max_sample_length
        print('CPSC2018Dataset initialized\nNumber of samples: {}\nUnique classes: {}'.format(self.__len__(), self.classes))

    def __len__(self):
        return len(self.path_labels)

    #Due to differences in sample lengths, now just extract first 5000 samples
    def __getitem__(self, idx):
        item = scipy.io.loadmat(self.generate_path(idx))
        waveform = self.convert_from_12_to_8_lead(item['ECG']['data'][0][0])
        length = waveform.shape[1]
        if self.max_sample_length:
            length = np.min([waveform.shape[1], self.max_sample_length])
            waveform_padded = np.zeros((waveform.shape[0], self.max_sample_length))
            waveform_padded[:, 0:length] = waveform[:, 0:length]
            waveform = waveform_padded
        gender = 0 if item['ECG']['sex'][0][0][0][0] == 'M' else 1
        age = item['ECG']['age'][0][0][0][0]
        label = self.path_labels['First_label'].iloc[idx] - 1 # Make label zero-indexed
        id = self.path_labels['id'].iloc[idx]
        
        if self.OOD_classname != 'none':                     
            sample = {'waveform': waveform, 'label': label, 'age': age, 'gender': gender, 'id': id, 'OOD_class': self.OOD_classname}
        else:
            sample = {'waveform': waveform, 'label': label, 'age': age, 'gender': gender, 'id': id}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    # Remove leads [III, aVR, aVL, aVF] from waveform
    def convert_from_12_to_8_lead(self, waveform):
        return np.delete(waveform, [2, 3, 4, 5], axis=0)

    # # Generate the path to the waveform
    def generate_path(self, idx):
        return "{}/{}.mat".format(self.waveform_dir, self.path_labels['Recording'].iloc[idx])

    # Plots the waveform of the specified lead indices.
    def plot_waveform(self, sample, lead_idx=None, interval=(0, 10000)):
        if lead_idx != None:
            fig, axs = plt.subplots(len(lead_idx), 1)
            for i, lead_ix in enumerate(lead_idx):
                axs[i].plot(sample['waveform'][lead_ix-1][interval[0]:interval[1]])
                # plt.xlabel('Samples @ {} hz'.format(sample['header']['sample_Fs']))
                axs[i].set_ylabel('Lead {}'.format(lead_ix))
        else:
            fig, axs = plt.subplots(sample['waveform'].shape[0], 1)
            for i, ax in enumerate(axs):
                ax.plot(sample['waveform'][i][interval[0]:interval[1]])
                # plt.xlabel('Samples @ {} hz'.format(sample['header']['sample_Fs']))
                ax.set_ylabel('Lead {}'.format(i + 1))
        # fig.suptitle('ECG: {}, Label: {}'.format(sample['header']['ptID'], sample['header']['label']),fontsize=12)
        plt.show()

    # Find unique number of classes  
    def get_classes(self):
        classes = self.path_labels['First_label'].unique()

        return sorted(classes) 
        
        # return sorted(classes)


