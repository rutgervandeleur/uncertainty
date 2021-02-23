from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
import json
import sys, os
import unittest

# Add project directory to path
testdir = os.path.dirname(__file__)
srcdir = '../project'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
from systems.ecgresnet_uncertainty import ECGResNetUncertaintySystem
from systems.ecgresnet_auxout import ECGResNetAuxOutSystem
from systems.ecgresnet_mcdropout import ECGResNetMCDropoutSystem
from systems.ecgresnet_ensemble import ECGResNetEnsembleSystem
from systems.ecgresnet_ssensemble import ECGResNetSnapshotEnsembleSystem
from systems.ecgresnet_varinf import ECGResNetVariationalInferenceSystem
from systems.ecgresnet_varinf_bayesdecomp import ECGResNetVariationalInference_BayesianDecompositionSystem
from systems.ecgresnet_ensemble_auxout import ECGResNetEnsemble_AuxOutSystem
from systems.ecgresnet_ssensemble_auxout import ECGResNetSnapshotEnsemble_AuxOutSystem
from systems.ecgresnet_mcdropout_auxout import ECGResNetMCDropout_AuxOutSystem

class TestModelInits(unittest.TestCase):
    def test_ensemble_init(self):
        seed_everything(1234)
        parser, ECGResNet_params = get_args()
        parser = ECGResNetEnsembleSystem.add_model_specific_args(parser)
        args = parser.parse_args()
        merged_dict = {**vars(args), **ECGResNet_params}

        model = ECGResNetEnsembleSystem(**merged_dict)

        self.assertTrue(isinstance(model, ECGResNetEnsembleSystem))

    def test_auxout_init(self):
        seed_everything(1234)
        parser, ECGResNet_params = get_args()
        parser = ECGResNetAuxOutSystem.add_model_specific_args(parser)
        args = parser.parse_args()
        merged_dict = {**vars(args), **ECGResNet_params}

        model = ECGResNetAuxOutSystem(**merged_dict)

        self.assertTrue(isinstance(model, ECGResNetAuxOutSystem))

    def test_mcdropout_init(self):
        seed_everything(1234)
        parser, ECGResNet_params = get_args()
        parser = ECGResNetMCDropoutSystem.add_model_specific_args(parser)
        args = parser.parse_args()
        merged_dict = {**vars(args), **ECGResNet_params}

        model = ECGResNetMCDropoutSystem(**merged_dict)

        self.assertTrue(isinstance(model, ECGResNetMCDropoutSystem))

    def test_ssensemble_init(self):
        seed_everything(1234)
        parser, ECGResNet_params = get_args()
        parser = ECGResNetSnapshotEnsembleSystem.add_model_specific_args(parser)
        args = parser.parse_args()
        merged_dict = {**vars(args), **ECGResNet_params}

        model = ECGResNetSnapshotEnsembleSystem(**merged_dict)

        self.assertTrue(isinstance(model, ECGResNetSnapshotEnsembleSystem))

    def test_varinf_init(self):
        seed_everything(1234)
        parser, ECGResNet_params = get_args()
        parser = ECGResNetVariationalInferenceSystem.add_model_specific_args(parser)
        args = parser.parse_args()
        merged_dict = {**vars(args), **ECGResNet_params}
        merged_dict['train_dataset_size'] = 1337 # arbitrary number for init
        merged_dict['val_dataset_size'] = 1337 # arbitrary number for init

        model = ECGResNetVariationalInferenceSystem(**merged_dict)

        self.assertTrue(isinstance(model, ECGResNetVariationalInferenceSystem))

    def test_ensemble_auxout_init(self):
        seed_everything(1234)
        parser, ECGResNet_params = get_args()
        parser = ECGResNetEnsemble_AuxOutSystem.add_model_specific_args(parser)
        args = parser.parse_args()
        merged_dict = {**vars(args), **ECGResNet_params}
        merged_dict['train_dataset_size'] = 1337 # arbitrary number for init
        merged_dict['val_dataset_size'] = 1337 # arbitrary number for init

        model = ECGResNetEnsemble_AuxOutSystem(**merged_dict)

        self.assertTrue(isinstance(model, ECGResNetEnsemble_AuxOutSystem))

    def test_ssensemble_auxout_init(self):
        seed_everything(1234)
        parser, ECGResNet_params = get_args()
        parser = ECGResNetSnapshotEnsemble_AuxOutSystem.add_model_specific_args(parser)
        args = parser.parse_args()
        merged_dict = {**vars(args), **ECGResNet_params}
        merged_dict['train_dataset_size'] = 1337 # arbitrary number for init
        merged_dict['val_dataset_size'] = 1337 # arbitrary number for init

        model = ECGResNetSnapshotEnsemble_AuxOutSystem(**merged_dict)

        self.assertTrue(isinstance(model, ECGResNetSnapshotEnsemble_AuxOutSystem))

    def test_ssensemble_auxout_init(self):
        seed_everything(1234)
        parser, ECGResNet_params = get_args()
        parser = ECGResNetMCDropout_AuxOutSystem.add_model_specific_args(parser)
        args = parser.parse_args()
        merged_dict = {**vars(args), **ECGResNet_params}
        merged_dict['train_dataset_size'] = 1337 # arbitrary number for init
        merged_dict['val_dataset_size'] = 1337 # arbitrary number for init

        model = ECGResNetMCDropout_AuxOutSystem(**merged_dict)

        self.assertTrue(isinstance(model, ECGResNetMCDropout_AuxOutSystem))

    def test_ssensemble_auxout_init(self):
        seed_everything(1234)
        parser, ECGResNet_params = get_args()
        parser = ECGResNetVariationalInference_BayesianDecompositionSystem.add_model_specific_args(parser)
        args = parser.parse_args()
        merged_dict = {**vars(args), **ECGResNet_params}
        merged_dict['train_dataset_size'] = 1337 # arbitrary number for init
        merged_dict['val_dataset_size'] = 1337 # arbitrary number for init

        model = ECGResNetVariationalInference_BayesianDecompositionSystem(**merged_dict)

        self.assertTrue(isinstance(model, ECGResNetVariationalInference_BayesianDecompositionSystem))

def get_args():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    temp_args, _ = parser.parse_known_args()
    file = open('project/ecgresnet_config.json', 'r')
    ECGResNet_params = json.load(file)
    file.close()

    return parser, ECGResNet_params

if __name__=='__main__':
    unittest.main()

