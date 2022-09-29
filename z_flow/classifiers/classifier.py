import math
import threading
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
from brainflow import BoardShim, BrainFlowPresets
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


class ClfError(Exception):
    pass


class Classifier:
    """Implements a classifier that holds a model, collect's samples,
    trains and predicts the model, and returns potential scores.

    :param board_shim: Brainflow BoardShim to collect data from EEG devices.
    :type board_shim: BoardShim
    :param model_type: Model type
    :type model_type: str
    :param time_range: The time range of data to collect when creating samples
    :type time_range: list[int]
    :param filter_freq_cutoff: Lower- and upper-bound filter cutoff frequencies, respectively.
    :type filter_freq_cutoff: list[float]
    :param method: Method to use when collecting samples.
    :type method: str
    """

    def __init__(self, board_shim: BoardShim, model_type: str, time_range: list[int],
                 filter_freq_cutoff: list[float], method: str) -> None:
        self.board_shim = board_shim
        self.model = self._create_model(model_type)
        self.time_range = time_range
        self.filter_freq_cutoff = filter_freq_cutoff
        self.method = self._cast_method(method)

        self.lock = threading.Lock()
        self.train_x = []
        self.train_y = []
        self.scores = {}

        # set some stuff
        self.window_size = 50  # in ms
        self.baseline_timeframe = 200  # in ms
        self.board_id = self.board_shim.get_board_id()

        self.eeg_preset = BrainFlowPresets.DEFAULT_PRESET
        self.eeg_sample_rate = self.board_shim.get_sampling_rate(self.board_id, self.eeg_preset)
        self.eeg_data_channels = board_shim.get_eeg_channels(self.board_id, self.eeg_preset)
        self.eeg_timestamp_channel = board_shim.get_timestamp_channel(self.board_id, self.eeg_preset)

        self.motion_preset = BrainFlowPresets.AUXILIARY_PRESET
        self.motion_sample_rate = self.board_shim.get_sampling_rate(self.board_id, self.motion_preset)
        self.motion_data_channels = board_shim.get_accel_channels(self.board_id, self.motion_preset) + \
            board_shim.get_gyro_channels(self.board_id, self.motion_preset)
        self.motion_timestamp_channel = board_shim.get_timestamp_channel(self.board_id, self.motion_preset)

        # compute some stuff
        self.wait_time = self.time_range[1] + 100  # Add additional 100ms waiting time to be safe
        self.total_event_duration = self.time_range[1] - self.time_range[0]
        # Add 100ms + 100ms to capture enough including the 100ms additional wait time.
        self.eeg_num_samples = math.ceil((self.total_event_duration + 200) / 1000 * self.eeg_sample_rate)
        self.motion_num_samples = math.ceil((self.total_event_duration + 200) / 1000 * self.motion_sample_rate)

    def _cast_method(self, method: str) -> any:
        if method == 'windowed-average-EEG':
            return self._window_averaged_eeg
        elif method == 'windowed-average-EEG-motion':
            return self._window_averaged_eeg_motion
        else:
            raise ClfError("Unknown collection method")

    def _create_model(self, model_type: str) -> any:
        if model_type == 'svm':
            return SVC()
        elif model_type == 'lda':
            return LinearDiscriminantAnalysis()
        else:
            raise ClfError("Unknown model_type")

    #-------------------------#
    # data collection methods #
    #-------------------------#

    def collect_sample(self, label: int | None, event_timestamp: float) -> None | npt.NDArray[np.float64]:
        """Collects sample data using the collection method and parameters provided when creating the model.
        If a label is given the sample is considered train data and stored as such internally.
        If label is None the sample is considered target data and X data is returned.

        :param label: Label, when None the sample is considered target data
        :type label: int | None
        :param event_timestamp: Original event timestamp
        :type event_timestamp: float
        :return: Returns None when collecting train samples, returns X data when collecting a target sample
        :rtype: None | npt.NDArray[np.float64]
        """
        x_data = self.method(event_timestamp)
        if label is None:  # predicting, return X data
            return x_data
        else:  # training, save X and y data, return nothing
            with self.lock:
                self.train_x.append(x_data)
                self.train_y.append(label)
            return None

    def _window_averaged_eeg_motion(self, event_timestamp: float) -> npt.NDArray[np.float64]:
        """Internal wrapper for `_window_averaged_eeg`, sets use_motion to True.

        :param event_timestamp: Original event timestamp
        :type event_timestamp: float
        :return: Returns X data
        :rtype: npt.NDArray[np.float64]
        """
        return self._window_averaged_eeg(event_timestamp, use_motion=True)

    def _window_averaged_eeg(self, event_timestamp: float, use_motion: bool = False) -> npt.NDArray[np.float64]:
        """Collects X data using the `window_averaged_eeg` method.
        Also implements `window_averaged_eeg`, by setting use_motion.
        Uses a window size of self.window_size, in ms.
        Re-references EEG data and applies baseline calculation as post-processing step

        :param event_timestamp: Original event timestamp
        :type event_timestamp: float
        :param use_motion: Allows using motion data, defaults to False
        :type use_motion: bool, optional
        :return: Returns X data
        :rtype: npt.NDArray[np.float64]
        """
        time.sleep(self.wait_time / 1000)  # wait_time is in ms
        data_eeg = self.board_shim.get_current_board_data(self.eeg_num_samples, self.eeg_preset).T
        if use_motion:
            data_motion = self.board_shim.get_current_board_data(self.motion_num_samples, self.motion_preset).T

        eeg_df = pd.DataFrame(data_eeg[:, self.eeg_data_channels], index=pd.to_datetime(
            data_eeg[:, self.eeg_timestamp_channel], unit='s'))
        if use_motion:
            motion_df = pd.DataFrame(data_motion[:, self.motion_data_channels],
                                     index=pd.to_datetime(data_motion[:, self.motion_timestamp_channel], unit='s'))

        # Use a Butterworth filter for EEG data
        Wn = np.array(self.filter_freq_cutoff) / self.eeg_sample_rate * 2
        b, a = signal.butter(5, Wn, btype='bandpass')
        eeg_df.loc[:, :] = signal.lfilter(b, a, eeg_df.to_numpy().T).T

        # re-reference EEG data
        eeg_mean = eeg_df.mean(axis=1)
        eeg_df['negative_mean'] = 0.0
        eeg_df = eeg_df.apply(lambda x: x - eeg_mean)  # subtract channel means from each channel

        # Baseline calculation
        event_timestamp = pd.to_datetime(event_timestamp, unit='s')
        event_start = event_timestamp + pd.Timedelta(milliseconds=self.time_range[0])
        # Add an additional window_size of milliseconds so downsampling has enough data to compute means.
        event_end = event_timestamp + pd.Timedelta(milliseconds=self.time_range[1] + self.window_size)
        baseline_end = event_start + pd.Timedelta(milliseconds=self.baseline_timeframe)

        baseline_mean = eeg_df.loc[event_start:event_end].mean(axis=0)
        eeg_df = eeg_df.loc[baseline_end:event_end] - baseline_mean

        # Downsample and flatten
        eeg_df = eeg_df.resample(f'{self.window_size}ms', origin=event_timestamp).mean()
        x_data = eeg_df.to_numpy().flatten()
        if use_motion:
            motion_df = motion_df.loc[baseline_end:event_end]
            motion_df = motion_df.resample(f'{self.window_size}ms', origin=event_timestamp).mean()
            x_data = np.concatenate([x_data, motion_df.to_numpy().flatten()])
        return x_data

    #---------------#
    # Model methods #
    #---------------#

    def train(self, use_cv: bool = True, n_folds: int = 5) -> dict:
        """Trains model given as model_type on object instantiation.
        By defaults uses a cross validation (CV) technique to compute scores.
        If use_cv is False, scores are only computed over train data.

        :param use_cv: Use cross validation to train model, defaults to True
        :type use_cv: bool, optional
        :param n_folds: Number of CV folds, defaults to 5
        :type n_folds: int, optional
        :return: Returns train (and test) scores.
        :rtype: dict
        """
        with self.lock:  # lock to prevent race condition
            train_x = np.array(self.train_x)
            train_y = np.array(self.train_y)

        if len(train_y) < 1:
            raise ClfError("No samples collected yet.")

        try:
            self.model.fit(train_x, train_y)
            if use_cv is True:
                self.scores = cross_validate(self.model, train_x, train_y, cv=n_folds,
                                             scoring=['precision', 'recall', 'f1', 'accuracy'],
                                             return_train_score=True)
            else:
                train_pred = self.model.predict(train_y)
                self.scores['train_precision'] = precision_score(train_y, train_pred)
                self.scores['train_recall'] = recall_score(train_y, train_pred)
                self.scores['train_f1'] = f1_score(train_y, train_pred)
                self.scores['train_accuracy'] = accuracy_score(train_y, train_pred)
        except ValueError as e:
            raise ClfError(e)

        return self.scores

    def predict(self, event_timestamp: float) -> list:
        """Collects and predicts single event sample given by `event_timestamp`.

        :param event_timestamp: Original event timestamp
        :type event_timestamp: float
        :raises ClfError: Passes on sklearn.exception.NotFittedError as ClfError
        :return: Returns a list containing the sample prediction, the probabilities, the log probabilities,
                 and possible classes.
        :rtype: list
        """
        try:
            check_is_fitted(self.model)
        except NotFittedError as e:
            raise ClfError(e)

        target_x = self.collect_sample(None, event_timestamp).reshape((1, -1))
        target_pred = self.model.predict(target_x)
        target_distance = self.model.decision_function(target_x)
        return target_pred, target_distance
