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
from utils_.helpers import *
import scipy
import scipy.io
from scipy import signal
import sys
import random
from scipy import misc
import matplotlib.pyplot as plt

################################################################################

class ECGDataset(Dataset):
    def __init__(self, path_labels_csv, waveform_dir, OOD_classname, label_column, transform=None, assert_shape=False, get_age=True, get_gender=True):
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
            filepath = os.path.join(self.waveform_dir, generate_path(self.path_labels['PseudoID'].iloc[idx]), str(self.path_labels['TestID'].iloc[idx])) + '.npy'
            if os.path.isfile(filepath):

                waveform = np.load(filepath)
                if self.path_labels['PseudoID'].iloc[idx] == 1:
                    waveform *= (random.randrange(70,130) / 100)
                    waveform = np.roll(waveform, random.randrange(100, 1000)) 
                if assert_shape==True:
                    assert waveform.shape == (8, 5000)

                    
                label = self.path_labels[self.label_column].iloc[idx]
                age = self.path_labels['Age'].iloc[idx] if self.get_age else 0
                gender = self.path_labels['Gender'].iloc[idx] if self.get_gender else 0
                id = self.path_labels['TestID'].iloc[idx]
            
            else:
                print("error in file path")
                print(filepath)
                print(self.path_labels['PseudoID'].iloc[idx])
                return
                
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

    # Due to differences in sample lengths, now just extract first 5000 samples
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
                axs[i].set_ylabel('Lead {}'.format(lead_ix))
        else:
            fig, axs = plt.subplots(sample['waveform'].shape[0], 1)
            for i, ax in enumerate(axs):
                ax.plot(sample['waveform'][i][interval[0]:interval[1]])
                ax.set_ylabel('Lead {}'.format(i + 1))
        plt.show()

    # Find unique number of classes  
    def get_classes(self):
        classes = self.path_labels['First_label'].unique()

        return sorted(classes) 
