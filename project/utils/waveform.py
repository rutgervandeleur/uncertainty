import numpy as np


def median_cutoff_points(ventricular_rate, ponset, toffset):
    """Calculate the median cutoff start and end points"""
    ponset = 0 if np.isnan(ponset) else int(ponset)
    toffset = 600 if np.isnan(toffset) else int(toffset)
    # limit the onset and offset to be in the range of 0-600
    # take some margin of 10ms (5*2) on the start and end indices
    margin = 5
    ponset = max(ponset - margin, 0)
    toffset = min(toffset, 600)
    if np.isnan(ventricular_rate):
        end = 600
    else:
        # calculate the average number of points between the QRS complexes
        rr_interval = (1 * 60 * 1000 / 2) / ventricular_rate
        # say that the end of a beat would be around the onset of the P wave
        # plus the avg. duration of one beat
        end = min(ponset + margin + rr_interval, 600)
        # if the GE measured T wave offset is larger than our calculated beat
        # endpoint, take the measured T wave offset
        end = max(end, toffset)
        if not np.isinf(end):
            end = int(end)
    return ponset, end


def median_cutoff(raw_wvf, ventricular_rate, ponset, toffset, repeat=False,
                  ones_padding=False, nans_padding=False):
    """Cutoff the values before the P-wave onset and T-wave offset of the
    median beat to decrease noise.
    """
    ponset = 0 if np.isnan(ponset) else int(ponset)
    ponset = 0 if ponset > raw_wvf.shape[1] else ponset
    toffset = 600 if np.isnan(toffset) else int(toffset)
    # limit the onset and offset to be in the range of 0-600
    # take some margin of 10ms (5*2) on the start and end indices
    margin = 5
    ponset = max(ponset - margin, 0)
    toffset = min(toffset, 600)
    if np.isnan(ventricular_rate):
        end = 600
    else:
        # calculate the average number of points between the QRS complexes
        rr_interval = int((1 * 60 * 1000 / 2) / ventricular_rate)
        # say that the end of a beat would be around the onset of the P wave
        # plus the avg. duration of one beat
        end = min(ponset + margin + rr_interval, 600)
        # if the GE measured T wave offset is larger than our calculated beat
        # endpoint, take the measured T wave offset
        end = max(end, toffset)
    cutoff_ecg = np.ones(raw_wvf.shape) if ones_padding else np.zeros(raw_wvf.shape)
    if nans_padding:
        cutoff_ecg[:] = np.nan
    cut_ecg = raw_wvf[:, ponset:end]
    cutoff_ecg[:, ponset:end] = cut_ecg
    if repeat:
        # prepend and append repeated values for each lead
        for lead in range(raw_wvf.shape[0]):
            # repeat first value before p wave onset
            cutoff_ecg[lead, 0:ponset] = np.full(ponset, cut_ecg[lead, 0])
            # repeat last value after beat endpoint
            cutoff_ecg[lead, end:600] = np.full(600 - end, cut_ecg[lead, -1])
    return cutoff_ecg
