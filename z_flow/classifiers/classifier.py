import time
from abc import ABC, abstractmethod

import numpy as np
from scipy import signal
from brainflow import BrainFlowPresets


class Classifier(ABC):
    def __init__(self, type, interval, filter, method):
        self.model = {}
        self.feature_list = []
        self.label_list = []
        self.type = type
        self.interval = interval
        self.filter = filter
        self.method = method

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, board_shim, board_id, event_timestamp):
        raise NotImplementedError

    def collect_sample(self, board_shim, board_id, event_timestamp):
        interval = self.interval
        filter = self.filter
        method = self.method

        # fetch data
        [interval_start, interval_end] = interval.split(',')
        interval_start = int(interval_start.replace("[", ""))
        interval_end = int(interval_end.replace("]", ""))
        fs = board_shim.get_sampling_rate(board_id, BrainFlowPresets.DEFAULT_PRESET)
        num_sample = int(fs*(interval_end-interval_start + 100)/1000)
        chan_timestamp = board_shim.get_timestamp_channel(board_id, BrainFlowPresets.DEFAULT_PRESET)
        chan_eeg = board_shim.get_eeg_channels(board_id, BrainFlowPresets.DEFAULT_PRESET)
        # await asyncio.sleep(interval_end/1000 + 0.1)
        time.sleep(interval_end/1000 + 0.1)
        data = board_shim.get_current_board_data(num_sample, BrainFlowPresets.DEFAULT_PRESET).T
        # calculate the real interval based on timestamp of event
        interval_real = (data[:, chan_timestamp] - event_timestamp)*1000
        # print([interval_real[0], interval_real[-1]])

        # preprocessing for EEG data
        data_eeg = data[:, chan_eeg]
        # filtering for EEG data
        if filter == '':
            pass
        else:
            [low, high] = filter.split(',')
            low = float(low.replace("[", ""))
            high = float(high.replace("]", ""))
            # here use butterworth band-pass filter
            Wn = np.array([low, high]) / fs * 2
            b, a = signal.butter(5, Wn, btype='bandpass')
            data_eeg = signal.lfilter(b, a, data_eeg.T).T

        # re-reference EEG data
        eeg_mean = np.mean(data_eeg, axis=1, keepdims=True)
        data_eeg = np.hstack([data_eeg, np.zeros((data_eeg.shape[0], 1))])  # add a new channel full of zeros
        data_eeg = data_eeg - eeg_mean

        # use [, -200] to do baseline for EEG data
        idx_bl = (interval_real < -200)
        idx_post = np.logical_and(interval_real > 0, interval_real < interval_end)
        data_eeg = data_eeg[idx_post, :] - np.mean(data_eeg[idx_bl, :], axis=0)
        interval_real = interval_real[idx_post]

        # feature extracting
        if method == '':
            pass
        elif method == 'windowed-average-EEG':
            data_extract = data_eeg
            win_len = 50  # window length in ms
            n_win = int(interval_end / win_len)
            feature_vector = []
            for i in range(n_win):
                idx_in_win = np.logical_and(interval_real > win_len*i, interval_real < win_len*(i+1))
                feature_vector.append(np.mean(data_extract[idx_in_win, :], axis=0))
            feature_vector = np.concatenate(np.squeeze(feature_vector))
            return feature_vector
        elif method == 'windowed-average-EEG-motion':
            chan_motion = board_shim.get_accel_channels(board_id) + board_shim.get_gyro_channels(board_id)
            data = data[idx_post, :]
            data_motion = data[:, chan_motion]
            data_extract = np.concatenate((data_eeg, data_motion), axis=1)
            win_len = 50  # window length in ms
            n_win = int(interval_end / win_len)
            feature_vector = []
            for i in range(n_win):
                idx_in_win = np.logical_and(interval_real > win_len*i, interval_real < win_len*(i+1))
                feature_vector.append(np.mean(data_extract[idx_in_win, :], axis=0))
            feature_vector = np.concatenate(np.squeeze(feature_vector))
            return feature_vector
        else:
            print('wrong keyword for <method>!')
