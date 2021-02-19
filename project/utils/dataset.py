import os
from torch.utils.data import Dataset
import scipy
import numpy as np
from .helpers import generate_path


class UniversalECGDataset(Dataset):
    """Universal ECG dataset in PyTorch Dataset format.
    """

    def __init__(self, dataset_type, waveform_dir, dataset, transform=None,
                 label='Label'):
        """Initializes the UMCU ECG datasets.

        Args:
            dataset_type (str): Type of dataset, options are 'umcu',
                'universal' and 'physionet'. UMCU and universal datasets
                contain numpy files, while the physionet dataset contains
                matlab files.
            waveform_dir (str): Path of the folder with the raw waveform files
            dataset (pd.DataFrame): Pandas DataFrame with the dataset your are
                using. Minimally required columns for UMCU are: PseudoID,
                TestID, SampleBase and Gain. For universal and physionet
                datasets we need Filename, SampleBase and Gain.
            transform (list): List of transformations.
            label (str): Name of the y variable in the dataset.
        """
        assert dataset_type in ['umcu', 'universal', 'physionet']

        if (('PseudoID' in dataset and 'TestID' in dataset)
           or 'Filename' in dataset):
            self.dataset = dataset
        else:
            print(('Please provide either a PseudoID/TestID combination or'
                   'a Filename in the dataset.'))
            raise

        self.waveform_dir = waveform_dir
        self.transform = transform
        self.label = label
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset_type == 'umcu':
            waveform = np.load(os.path.join(
                self.waveform_dir,
                generate_path(self.dataset['PseudoID'].iloc[idx]),
                '{}.npy'.format(str(self.dataset['TestID'].iloc[idx])),
            ))
            sample_id = self.dataset['TestID'].iloc[idx]

        elif self.dataset_type == 'universal':
            waveform = np.load(os.path.join(
                self.waveform_dir,
                '{}.npy'.format(str(self.dataset['Filename'].iloc[idx])),
            ))
            sample_id = self.dataset['Filename'].iloc[idx]

        elif self.dataset_type == 'physionet':
            waveform = scipy.io.loadmat(os.path.join(
                self.waveform_dir,
                '{}.mat'.format(str(self.dataset['Filename'].iloc[idx])),
            ))
            sample_id = self.dataset['Filename'].iloc[idx]

        # Add waveform, original sample base, gain and ID to sample
        sample = {
            'waveform': waveform,
            'samplebase': int(self.dataset['SampleBase'].iloc[idx]),
            'gain': float(self.dataset['Gain'].iloc[idx]),
            'id': sample_id,
        }

        # Sometimes additional information is needed (e.g. for a median cutoff)
        possible_cols = ['AcqDate', 'POnset', 'TOffset', 'VentricularRate',
                         'QOnset', 'POffset', 'QOffset', 'start_idx',
                         'end_idx']
        for col in possible_cols:
            if col in self.dataset:
                sample[col.lower()] = self.dataset[col].iloc[idx]

        if self.label in self.dataset.columns.values:
            label = self.dataset[self.label].iloc[idx]
            sample['label'] = label

        if self.transform:
            sample = self.transform(sample)

        return sample
