from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
import json
import sys, os
import unittest

# Add project directory to path
testdir = os.path.dirname(__file__)
srcdir = '../project'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
from systems.ecgresnet_ensemble import ECGResNetEnsembleSystem

class TestModelInits(unittest.TestCase):
    def test_ensemble_init(self):
        seed_everything(1234)
        parser, ECGResNet_params = get_args()
        parser = ECGResNetEnsembleSystem.add_model_specific_args(parser)
        args = parser.parse_args()
        merged_dict = {**vars(args), **ECGResNet_params}

        model = ECGResNetEnsembleSystem(**merged_dict)

        self.assertTrue(isinstance(model, ECGResNetEnsembleSystem))

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

