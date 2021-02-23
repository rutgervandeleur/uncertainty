import sys
sys.path.append('..')
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

# from pytorch_lightning.loggers.neptune import NeptuneLogger

import json
import pandas as pd

from utils.dataset import UniversalECGDataset
from systems.ecgresnet_uncertainty import ECGResNetUncertaintySystem
from systems.ecgresnet_auxout import ECGResNetAuxOutSystem
from systems.ecgresnet_mcdropout import ECGResNetMCDropoutSystem
from systems.ecgresnet_ensemble import ECGResNetEnsembleSystem
from systems.ecgresnet_ssensemble import ECGResNetSnapshotEnsembleSystem
from systems.ecgresnet_varinf import ECGResNetVariationalInferenceSystem
from systems.ecgresnet_ensemble_auxout import ECGResNetEnsemble_AuxOutSystem
from systems.ecgresnet_ssensemble_auxout import ECGResNetSnapshotEnsemble_AuxOutSystem
from systems.ecgresnet_mcdropout_auxout import ECGResNetMCDropout_AuxOutSystem
from utils.dataloader import CPSC2018Dataset
from utils.transforms import ToTensor, Resample
from utils.transforms import ApplyGain

def main(args, ECGResNet_params, model_class):

    # Merge config file with command-line arguments
    merged_dict = {**vars(args), **ECGResNet_params}
    print(args.model_name)

    transform = transforms.Compose([ToTensor(), ApplyGain(umc=False)])

    # Load train dataset
    if args.dataset == 'UMCU-Triage':
        dataset_params = json.load(open('configs/UMCU-Triage.json', 'r'))
        print('loaded dataset params')
    elif args.dataset == 'CPSC2018':
        dataset_params = json.load(open('configs/CPSC2018.json', 'r'))
        print('loaded dataset params')

        trainset = CPSC2018Dataset(path_labels_csv = dataset_params['train_labels_csv'],
                              waveform_dir = dataset_params['data_dir'],
                              OOD_classname = str(dataset_params['OOD_classname']),
                              transform = transform,
                              max_sample_length = dataset_params['max_sample_length'])

        validationset = CPSC2018Dataset(path_labels_csv = dataset_params['train_labels_csv'],
                              waveform_dir = dataset_params['data_dir'],
                              OOD_classname = str(dataset_params['OOD_classname']),
                              transform = transform,
                              max_sample_length = dataset_params['max_sample_length'])

        testset = CPSC2018Dataset(path_labels_csv = dataset_params['train_labels_csv'],
                              waveform_dir = dataset_params['data_dir'],
                              OOD_classname = str(dataset_params['OOD_classname']),
                              transform = transform,
                              max_sample_length = dataset_params['max_sample_length'])

    merged_dict['num_classes'] = dataset_params['num_classes']
    merged_dict['train_dataset_size'] = len(trainset)
    merged_dict['val_dataset_size'] = len(validationset)
    merged_dict['test_dataset_size'] = len(testset)

    # Initialize dataloaders
    train_loader = DataLoader(trainset, batch_size=ECGResNet_params['batch_size'], num_workers=8)
    val_loader = DataLoader(validationset, batch_size=ECGResNet_params['batch_size'], num_workers=8)
    test_loader = DataLoader(testset, batch_size=ECGResNet_params['batch_size'], num_workers=8)


    # Initialize model

    # Turn of automatic optimization when dealin with ensembling methods
    # if merged_dict['ensembling_method'] == True:
    #     ic('WEINTHEREEEE')
    #     merged_dict['automatic_optimization'] = False
    # ic(merged_dict)

    model = model_class(**merged_dict)
    print('Initialized {}'.format(model.__class__.__name__))

    # Initialize Logger
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')

    # Initialize trainer
    k = 1
    trainer = Trainer.from_argparse_args(args, max_epochs=ECGResNet_params['max_epochs'], logger=tb_logger, log_every_n_steps=k)

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Test model
    trainer.test(test_dataloaders=test_loader)

    # Save model
    model.save_results()

    # Now make prediction with model and return uncertainty
    tb_logger.save()


# Selects the correct model class based on provided 'epistemic_method' and
# 'aleatoric_method' arguments
def get_model_class(args):
    if temp_args.aleatoric_method == 'none':
        if temp_args.epistemic_method == 'ensemble':
            # ensemble_none
            return ECGResNetEnsembleSystem

        elif temp_args.epistemic_method == 'mcdropout':
            #mcdropout_none
            return ECGResNetMCDropoutSystem

        elif temp_args.epistemic_method == 'varinf':
            #varinf_none
            return ECGResNetVariationalInferenceSystem
        
        elif temp_args.epistemic_method == 'ssensemble':
            # ssensemble_none
            return ECGResNetSnapshotEnsembleSystem

        elif temp_args.epistemic_method == 'none':
            # none_none
            return ECGResNetUncertaintySystem

    elif temp_args.aleatoric_method == 'auxout':
        if temp_args.epistemic_method == 'none':
            # none_auxout
            return ECGResNetAuxOutSystem

        elif temp_args.epistemic_method == 'ensemble':
            # ensemble_auxout
            return ECGResNetEnsemble_AuxOutSystem

        elif temp_args.epistemic_method == 'mcdropout':
            # mcdropout_auxout
            return ECGResNetMCDropout_AuxOutSystem

        elif temp_args.epistemic_method == 'ssensemble':
            # ssensemble_auxout
            return ECGResNetSnapshotEnsemble_AuxOutSystem

    elif temp_args.aleatoric_method == 'bayes-decomp':
        # varinf_bayes-decomp
        return ECGResNetVariationalInference_BayesianDecompositionSystem


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument('--epistemic_method', type=str, default='none', help='mcdropout, ensemble, ssensemble, varinf, none')
    parser.add_argument('--aleatoric_method', type=str, default='none', help='aux-out, bayes-decomp, none')
    parser.add_argument('--dataset', type=str, default='CPSC2018', help='UMCU-Triage, CPSC2018')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # Get model class based on passed arguments
    model_class = get_model_class(temp_args)

    # Add arguments dependent on model class 
    parser = model_class.add_model_specific_args(parser)
        
    # Parse command-line arguments
    args = parser.parse_args()

    # Turn of automatic optimization when dealin with ensembling methods
    if args.ensembling_method == True:
        args.automatic_optimization = False

    # Get standard network parameters from config file
    ECGResNet_params = json.load(open('ecgresnet_config.json', 'r'))

    # train
    main(args, ECGResNet_params, model_class)
