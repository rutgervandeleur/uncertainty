from pytorch_lightning import Trainer, seed_everything
import ecg_uncertainty
from ecg_uncertainty.project.systems.ecgresnet_varinf import ECGResNetVariationalInferenceSystem

def test_lit_classifier():
    seed_everything(1234)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    temp_args, _ = parser.parse_known_args()

    parser = ECGResNetVariationalInferenceSystem.add_model_specific_args(parser)
    args = parser.parse_args()

    ECGResNet_params = json.load(open('ecgresnet_config.json', 'r'))

    merged_dict = {**vars(args), **ECGResNet_params}

    model = ECGResNetVariationalInferenceSystem(**merged_dict)
    assert model.model_name == 'varinf_none'
#     trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
#     trainer.fit(model, train, val)

#     results = trainer.test(test_dataloaders=test)
#     assert results[0]['test_acc'] > 0.7
