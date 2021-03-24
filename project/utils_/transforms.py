import torch
import numpy as np
from scipy import interpolate
from .waveform import median_cutoff


class ToTensor_old(object):
    """Converts ndarrays in sample to FloatTensors.
    """
    def __call__(self, sample):
        waveform = sample['waveform']
        if waveform.shape[0] == 8:
            sample['waveform'] = torch.from_numpy(waveform).type(torch.float)
        elif waveform.shape[0] == 12:
            # select only leads I, II and V1 until V6
            waveform = waveform[[0, 1, 6, 7, 8, 9, 10, 11], :]
            sample['waveform'] = torch.from_numpy(waveform).type(torch.float)
        else:
            raise NotImplementedError

        if 'label' in sample:
            sample['label'] = torch.from_numpy(
                # np.array(sample['label'])).type(torch.float)
                np.array(sample['label'])).type(torch.long)

        return sample

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        # waveform = sample['waveform'][0:8,]
        waveform = sample['waveform']
        sample['waveform'] = torch.from_numpy(waveform).type(torch.FloatTensor)
        sample['age'] = torch.from_numpy(np.array(sample['age'])).type(torch.FloatTensor)
        sample['gender'] = torch.from_numpy(np.array(sample['gender'])).type(torch.FloatTensor)
        return sample
 

# class ApplyGain(object):
#     """Normalize ECG signal by multiplying by specified gain and converting to
#     millivolts.
#     """
#     def __call__(self, sample):
#         sample['waveform'] *= sample['gain']

#         return sample

class ApplyGain(object):
    """
    Normalize ECG signal by multiplying by specified gain and converting to millivolts
    """
    def __init__(self, umc = True):
        self.umc = umc
        
    def __call__(self, sample, umc = True):
        if self.umc:
            waveform = sample['waveform'] * 0.001 * 4.88
        else:
            waveform = sample['waveform'] * 0.001
        sample['waveform'] = waveform
        
        return sample



class To12Lead(object):
    """Convert 8 lead waveforms to their 12 lead equivalent.
    """
    def __call__(self, sample):
        waveform = sample['waveform']

        if waveform.shape[0] != 12:
            out = np.zeros((12, waveform.shape[1]))
            # I and II
            out[0:2, :] = waveform[0:2, :] 
            # III = II - I
            out[2, :] = waveform[1, :] - waveform[0, :]
            # aVR = -(I + II)/2
            out[3, :] = -(waveform[0, :] + waveform[1, :]) / 2
            # aVL = I - II/2
            out[4, :] = waveform[0, :] - (waveform[1, :] / 2)
            # aVF = II - I/2
            out[5, :] = waveform[1, :] - (waveform[0, :] / 2)
            # V1 to V6
            out[6:12, :] = waveform[2:8, :]

            sample['waveform'] = out

        return sample


class Resample(object):
    """Convert 8 lead waveforms to their 12 lead equivalent using linear
    interpolation.
    """
    def __init__(self, sample_freq):
        """Initializes the resample transformation.

        Args:
            sample_freq (int): The required sampling frequency to resample to.
        """
        self.sample_freq = int(sample_freq)

    def __call__(self, sample):
        samplebase = int(sample['samplebase'])
        waveform = sample['waveform']

        if samplebase != self.sample_freq:
            length = int(waveform.shape[1])
            x = np.linspace(0, length / samplebase, num=length)
            f = interpolate.interp1d(x, waveform, axis=1)
            out_length = int((length / samplebase) * self.sample_freq)
            xnew = np.linspace(0, length / samplebase, 
                               num=out_length)
            sample['waveform'] = f(xnew)

        return sample


class MedianCutoff(object):
    """Cut off all noise before the P wave and after the T wave"""
    def __call__(self, sample):
        sample['waveform'] = median_cutoff(sample['waveform'],
                                           sample['ventricularrate'],
                                           sample['ponset'],
                                           sample['toffset']
                                           )

        return sample


# Transerred from old versions, need review
class SingleBeat(object):
    """Extract single beat from rhythm using provided start and end indices."""
    def __call__(self, sample):
        start_idx = int(sample['start_idx'])
        end_idx = int(sample['end_idx'])
        sample['waveform'] = sample['waveform'][:, start_idx:end_idx]

        return sample


class EmbedECG(object):
    """Encode ECG signal into its lower-dimensional encoding"""
    def __init__(self, Embedder, embedder_loc):
        self.embedder, _ = Embedder(embedder_loc)
        self.embedder = self.embedder.cpu()

    def __call__(self, sample):
        sample['original_waveform'] = sample['waveform']
        self.embedder.eval()
        with torch.no_grad():
            x = sample['waveform'].unsqueeze(dim=0)
            enc_mu, _ = self.embedder.encoder(x)
            z = enc_mu.squeeze(dim=0)
            sample['waveform'] = z

        return sample
