import time
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd
from brainflow import BoardIds, BoardShim, BrainFlowPresets
from scipy import signal


class Classifier(ABC):
    def __init__(self, model_type: str, interval: str, filter: str, method: str) -> None:
        self.model = {}
        self.feature_list = []
        self.label_list = []
        self.type = model_type
        self.interval = interval
        self.filter = filter
        self.method = method

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, board_shim: BoardShim, board_id: BoardIds, event_timestamp: float) -> None:
        raise NotImplementedError

    def collect_sample(self, board_shim: BoardShim, board_id: BoardIds, event_timestamp: float) -> npt.NDArray[np.float64]:
        interval = self.interval
        filter = self.filter
        method = self.method

        # fetch data
        [interval_start, interval_end] = interval.split(',')
        interval_start = int(interval_start.replace("[", ""))
        interval_end = int(interval_end.replace("]", ""))
        eeg_sample_rate = board_shim.get_sampling_rate(board_id, BrainFlowPresets.DEFAULT_PRESET)
        motion_sample_rate = board_shim.get_sampling_rate(board_id, BrainFlowPresets.AUXILIARY_PRESET)
        eeg_num_sample = int(eeg_sample_rate*(interval_end-interval_start + 100) / 1000)
        motion_num_sample = int(motion_sample_rate*(interval_end-interval_start + 100) / 1000)
        # await asyncio.sleep(interval_end / 1000 + 0.1)
        time.sleep(interval_end / 1000 + 0.1)
        data_eeg = board_shim.get_current_board_data(eeg_num_sample, BrainFlowPresets.DEFAULT_PRESET).T
        data_motion = board_shim.get_current_board_data(motion_num_sample, BrainFlowPresets.AUXILIARY_PRESET).T
        # calculate the real interval based on timestamp of event
        eeg_interval_real = (data_eeg[:, board_shim.get_timestamp_channel(
            board_id, BrainFlowPresets.DEFAULT_PRESET)] - event_timestamp) * 1000
        motion_interval_real = (data_motion[:, board_shim.get_timestamp_channel(
            board_id, BrainFlowPresets.AUXILIARY_PRESET)] - event_timestamp) * 1000
        # print([interval_real[0], interval_real[-1]])

        data_eeg = data_eeg[:, board_shim.get_eeg_channels(board_id, BrainFlowPresets.DEFAULT_PRESET)]
        data_motion = data_motion[:, board_shim.get_accel_channels(board_id, BrainFlowPresets.AUXILIARY_PRESET) + \
            board_shim.get_gyro_channels(board_id, BrainFlowPresets.AUXILIARY_PRESET)]

        # filtering for EEG data
        if filter == '':
            pass
        else:
            [low, high] = filter.split(',')
            low = float(low.replace("[", ""))
            high = float(high.replace("]", ""))
            # here use butterworth band-pass filter
            Wn = np.array([low, high]) / eeg_sample_rate * 2
            b, a = signal.butter(5, Wn, btype='bandpass')
            data_eeg = signal.lfilter(b, a, data_eeg.T).T

        # re-reference EEG data
        eeg_mean = np.mean(data_eeg, axis=1, keepdims=True)
        data_eeg = np.hstack([data_eeg, np.zeros((data_eeg.shape[0], 1))])  # add a new channel full of zeros
        data_eeg = data_eeg - eeg_mean

        # use [, -200] to do baseline for EEG data
        idx_bl = (eeg_interval_real < -200)
        idx_post = np.logical_and(eeg_interval_real > 0, eeg_interval_real < interval_end)
        data_eeg = data_eeg[idx_post, :] - np.mean(data_eeg[idx_bl, :], axis=0)
        eeg_interval_real = eeg_interval_real[idx_post]

        idx_bl = (motion_interval_real < -200)
        idx_post = np.logical_and(motion_interval_real > 0, motion_interval_real < interval_end)
        data_motion = data_motion[idx_post, :] # should we also extract the mean here?
        motion_interval_real = motion_interval_real[idx_post]

        # feature extracting
        if method == '':
            pass
        elif method == 'windowed-average-EEG':
            win_len = 50  # window length in ms
            n_win = int(interval_end / win_len)
            feature_vector = []
            for i in range(n_win):
                idx_in_win = np.logical_and(eeg_interval_real > win_len*i, eeg_interval_real < win_len*(i+1))
                feature_vector.append(np.mean(data_eeg[idx_in_win, :], axis=0))
            feature_vector = np.concatenate(np.squeeze(feature_vector))
            return feature_vector
        elif method == 'windowed-average-EEG-motion':
            win_len = 50  # window length in ms
            n_win = int(interval_end / win_len)
            feature_vector = []
            for i in range(n_win):
                idx_in_win_eeg = np.logical_and(eeg_interval_real > win_len*i, eeg_interval_real < win_len*(i+1))
                idx_in_win_motion = np.logical_and(motion_interval_real > win_len*i, motion_interval_real < win_len*(i+1))
                feature_vector.append(np.mean(data_eeg[idx_in_win_eeg, :], axis=0))
                feature_vector.append(np.mean(data_motion[idx_in_win_motion, :], axis=0))
            feature_vector = np.concatenate(np.squeeze(feature_vector))
            return feature_vector
        else:
            print('wrong keyword for <method>!')
