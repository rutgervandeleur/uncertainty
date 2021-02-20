from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
import json
import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../project'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

# import unittest
from systems.ecgresnet_ensemble import ECGResNetEnsembleSystem

def test_model_init():
    seed_everything(1234)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    temp_args, _ = parser.parse_known_args()

    parser = ECGResNetEnsembleSystem.add_model_specific_args(parser)
    args = parser.parse_args()

    ECGResNet_params = json.load(open('project/ecgresnet_config.json', 'r'))

    merged_dict = {**vars(args), **ECGResNet_params}

    model = ECGResNetEnsembleSystem(**merged_dict)

    assert isinstance(model, ECGResNetEnsembleSystem)

if __name__=='__main__':
    test_model_init()

