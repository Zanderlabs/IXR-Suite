import argparse
import logging
import sys
# f = open('test_board.log', 'w')
# sys.stdout = f
# print("test")
# f.close()

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import *
import time
import threading
from sklearn import svm
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock
from abc import ABC, abstractmethod


class Graph:
    def __init__(self, board_shim, calib_length, power_length, scale, offset, head_impact):
        pg.setConfigOption('background', '#264653')
        pg.setConfigOption('foreground', '#e9f5db')

        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.ppg_channels = BoardShim.get_ppg_channels(self.board_id)
        # print(BoardShim.get_board_descr(self.board_id))
        self.gyro_channels = BoardShim.get_gyro_channels(self.board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 100
        self.window_size_plot = 7
        self.window_size_compute = 1.5
        self.num_points_plot = self.window_size_plot * self.sampling_rate
        self.num_points_compute = int(self.window_size_compute * self.sampling_rate)

        self.timestamp_old = 0

        # # brainflow ML module
        # MLModel.release_all()
        # concentration_params = BrainFlowModelParams(BrainFlowMetrics.CONCENTRATION.value, BrainFlowClassifiers.KNN.value)
        # self.concentration = MLModel(concentration_params)
        # self.concentration.prepare()
        # relaxation_params = BrainFlowModelParams(BrainFlowMetrics.RELAXATION.value, BrainFlowClassifiers.REGRESSION.value)
        # self.relaxation = MLModel(relaxation_params)
        # self.relaxation.prepare()

        # selfmade power metrics
        self.calib_length = int(calib_length * 1000 / self.update_speed_ms)
        self.hist_length = int(power_length * 1000 / self.update_speed_ms)
        self.brain_scale = scale
        self.brain_center = offset
        self.head_impact = head_impact

        self.inverse_workload_calib = [0, 1]
        self.inverse_workload_hist = [0, 1]
        self.inverse_workload = 0
        self.engagement_calib = [0, 1]
        self.engagement_hist = [0, 1]
        self.engagement = 0
        self.power_metrics = 0

        # LSL stream
        info_transmit = StreamInfo('BrainPower', 'Z-metric', 1, 0, 'float32', 'zflow_transmit_power')
        self.outlet_transmit = StreamOutlet(info_transmit)

        # start GUI
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='Z-flow', size=(1500, 1000))

        self._init_pens()
        self._init_timeseries()
        self._init_psd()
        self._init_band_plot()
        self._init_brain_power_plot()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def _init_pens(self):
        self.pens = list()
        self.brushes = list()
        colors = ['#e9c46a', '#f4a261', '#e76f51', '#d62828', '#2a9d8f', '#168aad', '#e9f5db', '#A57E2F', '#A53B2F']
        for i in range(len(colors)):
            pen = pg.mkPen({'color': colors[i], 'width': 2})
            self.pens.append(pen)
            brush = pg.mkBrush(colors[i])
            self.brushes.append(brush)

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()

        axeslabels_eeg = ['left ear', 'left front', 'right front', 'right ear']
        for i in range(len(self.eeg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            p.setYRange(-150, 150, padding=0)
            p.showAxis('left', False)
            p.setTitle(axeslabels_eeg[i])
            self.plots.append(p)
            curve = p.plot(pen=self.pens[i % len(self.pens)])
            # curve.setDownsampling(auto=True, method='mean', ds=3)
            self.curves.append(curve)

        axeslabels_gyro = ['gyro 1', 'gyro 2', 'gyro 3']
        for i in range(len(self.gyro_channels)):
            p = self.win.addPlot(row=i + len(self.eeg_channels), col=0)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            p.setYRange(-250, 250, padding=0)
            p.showAxis('left', False)
            p.setTitle(axeslabels_gyro[i])
            self.plots.append(p)
            curve = p.plot(pen=self.pens[i % len(self.pens)])
            # curve.setDownsampling(auto=True, method='mean', ds=3)
            self.curves.append(curve)

        axeslabels_ppg = ['heart']
        p = self.win.addPlot(row=1 + len(self.eeg_channels) + len(self.gyro_channels), col=0)
        p.setMenuEnabled('left', False)
        p.showAxis('bottom', False)
        p.setMenuEnabled('bottom', False)
        p.setYRange(-1500, 2000, padding=0)
        p.showAxis('left', False)
        p.setTitle(axeslabels_ppg[0])
        self.plots.append(p)
        curve = p.plot(pen=self.pens[3])
        # curve.setDownsampling(auto=True, method='mean', ds=3)
        self.curves.append(curve)

    def _init_psd(self):
        self.psd_plot = self.win.addPlot(row=0, col=1, rowspan=4)
        self.psd_plot.showAxis('left', False)
        self.psd_plot.setMenuEnabled('left', False)
        self.psd_plot.setTitle('spectral power')
        self.psd_plot.setLogMode(False, True)
        self.psd_plot.setLabel('bottom', 'frequency (Hz)')
        self.psd_plot.setXRange(0, 50, padding=0)
        self.psd_curves = list()
        self.psd_size = DataFilter.get_nearest_power_of_two(self.sampling_rate)
        for i in range(len(self.eeg_channels)):
            psd_curve = self.psd_plot.plot(pen=self.pens[i % len(self.pens)])
            psd_curve.setDownsampling(auto=True, method='mean', ds=3)
            self.psd_curves.append(psd_curve)

    def _init_band_plot(self):
        self.band_plot = self.win.addPlot(row=4, col=1, rowspan=2)
        self.band_plot.showAxis('left', False)
        self.band_plot.setMenuEnabled('left', False)
        self.band_plot.showAxis('bottom', True)
        self.band_plot.setMenuEnabled('bottom', False)
        self.band_plot.setTitle('EEG band powers')
        y = [0, 0, 0, 0, 0]
        x = [1, 2, 3, 4, 5]
        self.band_bar = pg.BarGraphItem(x=x, height=y, width=0.8, pen=self.pens[4], brush=self.brushes[4])
        self.band_plot.addItem(self.band_bar)
        self.band_plot.setXRange(0.1, 5.9, padding=0)
        self.band_plot.setYRange(-0.1, 50, padding=0)
        ticklabels = ['', 'delta', 'theta', 'alpha', 'beta', 'gamma']
        tickdict = dict(enumerate(ticklabels))
        ay = self.band_plot.getAxis('bottom')
        ay.setTicks([tickdict.items()])

    def _init_brain_power_plot(self):

        self.power_plot = self.win.addPlot(row=6, col=1, rowspan=3)
        self.power_plot.setTitle('final brain power')

        self.power_plot.showAxis('left', False)
        self.power_plot.setMenuEnabled('left', False)
        self.power_plot.showAxis('bottom', True)
        self.power_plot.setMenuEnabled('bottom', False)
        y = [0]
        x = [1]
        self.power_bar = pg.BarGraphItem(x=x, height=y, width=0.8, pen=self.pens[5], brush=self.brushes[5])
        self.power_plot.addItem(self.power_bar)
        self.power_plot.setXRange(0.1, 1.9, padding=0)
        self.power_plot.setYRange(-0.1, 1.1, padding=0)
        ticklabels = ['', '']
        tickdict = dict(enumerate(ticklabels))
        ay = self.power_plot.getAxis('bottom')
        ay.setTicks([tickdict.items()])

    def update(self):
        try:
            data = self.board_shim.get_current_board_data(self.num_points_compute)
            data_plot = self.board_shim.get_current_board_data(self.num_points_plot)

            # re-reference
            eeg_data_plot = data_plot[self.eeg_channels, :]
            eeg_data_plot_mean = np.mean(eeg_data_plot, axis=0)
            eeg_data_plot = eeg_data_plot - eeg_data_plot_mean
            data_plot[self.eeg_channels, :] = eeg_data_plot

            eeg_data = data[self.eeg_channels, :]
            eeg_data_mean = np.mean(eeg_data, axis=0)
            eeg_data = eeg_data - eeg_data_mean
            data[self.eeg_channels, :] = eeg_data

            #  power_metrics = [0,0,0,0]

            # movement
            for count, channel in enumerate(self.gyro_channels):
                # plot timeseries
                self.curves[count + len(self.eeg_channels)].setData(data_plot[channel].tolist())

            head_movement = np.clip(np.mean(np.abs(data[self.gyro_channels])) / 50, 0, 1)
            #  power_metrics[2] = head_movement

            # heart
            channel = self.ppg_channels[0]
            DataFilter.detrend(data_plot[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_highpass(data_plot[channel], BoardShim.get_sampling_rate(self.board_id), 1.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_lowpass(data_plot[channel], BoardShim.get_sampling_rate(self.board_id), 40.0, 5,
                                       FilterTypes.CHEBYSHEV_TYPE_1.value, 1)
            self.curves[len(self.eeg_channels) + len(self.gyro_channels)].setData(data_plot[channel].tolist())

            # eeg processing
            avg_bands = [0, 0, 0, 0, 0]
            frontal_theta = 1
            parietal_alpha = 1
            engagement_idx = 1

            for count, channel in enumerate(self.eeg_channels):
                # prepare

                DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(data[channel], self.sampling_rate, 30.0, 58.0, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandstop(data[channel], self.sampling_rate, 50.0, 4.0, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)
                # preprocess plot
                DataFilter.detrend(data_plot[channel], DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(data_plot[channel], self.sampling_rate, 30.0, 56.0, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandstop(data_plot[channel], self.sampling_rate, 50.0, 4.0, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)
                # plot timeseries
                self.curves[count].setData(data_plot[channel].tolist())
                if data.shape[1] > self.psd_size:
                    # compute psd
                    psd_data = DataFilter.get_psd_welch(data[channel], self.psd_size, self.psd_size // 2,
                                                        self.sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
                    lim = min(48, len(psd_data[0]))
                    self.psd_curves[count].setData(psd_data[1][0:lim].tolist(), psd_data[0][0:lim].tolist())
                    # compute bands
                    delta = DataFilter.get_band_power(psd_data, 1.0, 4.0)
                    theta = DataFilter.get_band_power(psd_data, 4.0, 8.0)
                    alpha = DataFilter.get_band_power(psd_data, 8.0, 13.0)
                    beta = DataFilter.get_band_power(psd_data, 13.0, 30.0)
                    gamma = DataFilter.get_band_power(psd_data, 30.0, 60.0)
                    avg_bands[0] = avg_bands[0] + delta
                    avg_bands[1] = avg_bands[1] + theta
                    avg_bands[2] = avg_bands[2] + alpha
                    avg_bands[3] = avg_bands[3] + beta
                    avg_bands[4] = avg_bands[4] + gamma

                    # compute selfmade brain metrics
                    engagement_idx += (beta / (theta + alpha)) / gamma

                    if count == 1 or count == 4:
                        parietal_alpha += alpha / gamma
                    else:
                        frontal_theta += theta / gamma

            avg_bands = [int(x / len(self.eeg_channels)) for x in avg_bands]  # average bands were just sums

            engagement_idx = engagement_idx / 4
            parietal_alpha = parietal_alpha / 2
            frontal_theta = frontal_theta / 2

            # engagement
            self.engagement_calib.append(engagement_idx)
            if len(self.engagement_calib) > self.calib_length:
                del self.engagement_calib[0]

            if len(self.engagement_hist) > self.hist_length:
                del self.engagement_hist[0]

            # scale
            engagement_z = (engagement_idx - np.mean(self.engagement_calib)) / np.std(self.engagement_calib)
            engagement_z /= 2 * self.brain_scale
            engagement_z += self.brain_center
            engagement_z = np.clip(engagement_z, 0.05, 1)
            self.engagement_hist.append(engagement_z)
            # print(engagement_z)
            # print(self.engagement_hist)

            # weighted mean
            engagement_weighted_mean = 0
            sumweight = 0
            for count, hist_val in enumerate(self.engagement_hist):
                engagement_weighted_mean += hist_val * count
                sumweight += count

            engagement_weighted_mean = engagement_weighted_mean / sumweight

            self.engagement = engagement_weighted_mean

            #   # inverse workload
            #   inverse_workload_idx = parietal_alpha/frontal_theta
            #   self.inverse_workload_calib.append(inverse_workload_idx)
            #   if len(self.inverse_workload_calib) > self.calib_length:
            #       del self.inverse_workload_calib[0]
            #
            #   if len(self.inverse_workload_hist) > self.hist_length:
            #       del self.inverse_workload_hist[0]
            #
            # #  print('mean: ' + str(np.mean(self.inverse_workload_calib)))
            #  # print('std: ' + str(np.std(self.inverse_workload_calib)))
            #
            #   # scale
            #   inverse_workload_z = (inverse_workload_idx - np.mean(self.inverse_workload_calib)) / np.std(self.inverse_workload_calib)
            #   inverse_workload_z /= 2*self.brain_scale
            #   inverse_workload_z += self.brain_center
            #   inverse_workload_z = np.clip(inverse_workload_z,0.05,1)
            #   self.inverse_workload_hist.append(inverse_workload_z)
            #
            #   # weighted mean
            #   inverse_workload_weighted_mean = 0
            #   sumweight = 0
            #   for count, hist_val in enumerate(self.inverse_workload_hist):
            #       inverse_workload_weighted_mean += hist_val*count
            #       sumweight += count
            #
            #   inverse_workload_weighted_mean = inverse_workload_weighted_mean / sumweight
            #
            #   self.inverse_workload = inverse_workload_weighted_mean

            self.power_metrics = np.float32(self.engagement + (1 - head_movement) * self.head_impact)

            # power_metrics[3] = self.inverse_workload

            # # ML brain metrics
            # bands = DataFilter.get_avg_band_powers(data_plot, self.eeg_channels, self.sampling_rate, True)
            # feature_vector = np.concatenate((bands[0], bands[1]))
            #
            # power_metrics[0] = self.concentration.predict(feature_vector)
            #  power_metrics[1] = self.relaxation.predict(feature_vector)

            # plot bars
            self.band_bar.setOpts(height=avg_bands)
            self.power_bar.setOpts(height=self.power_metrics)

            # print('###################')
            # print(self.power_metrics)
            # print('###################')

            self.outlet_transmit.push_sample([self.power_metrics])

            self.app.processEvents()

        except:
            print('Could not get powers during update!')


class ClassifierFactory:
    def __init__(self, type):
        self.type = type

    def __call__(self, *args, **kwargs):
        if self.type == 'LDA':
            return LDA(*args, **kwargs)
        elif self.type == 'SVM':
            return SVM(*args, **kwargs)
        else:
            raise ValueError("Type not known!")


class Classifier(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, feature_list, label_list):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, board_shim, board_id, true_label):
        raise NotImplementedError


class LDA(Classifier):
    def __init__(self, feature_list, label_list):
        super().__init__()
        self.feature_list = feature_list
        self.label_list = label_list
        self.model = None

    def train(self, feature_list, label_list):
        X = np.array(feature_list).T  # features*samples
        y = np.array(label_list)
        mu1 = np.mean(X[:, y == 1], axis=1)
        mu0 = np.mean(X[:, y == 0], axis=1)
        # center features to estimate covariance
        Xpool = np.concatenate((X[:, y == 1] - mu1[:, np.newaxis], X[:, y == 0] - mu0[:, np.newaxis]), axis=1)
        C = np.cov(Xpool)
        w = np.linalg.pinv(C).dot(mu1 - mu0)
        b = w.T.dot((mu1 + mu0) / 2)
        self.model = [w, b]

    def predict(self, model, board_shim, board_id, true_label):
        sample_to_predict = np.array(collect_features(board_shim, board_id)).T
        result = model[0].T.dot(sample_to_predict) - model[1]
        print('prediction result:', result, 'true label:', true_label)


class SVM(Classifier):
    def __init__(self, feature_list, label_list):
        super().__init__()
        self.feature_list = feature_list
        self.label_list = label_list
        self.model = None

    def train(self, feature_list, label_list):
        X = np.array(feature_list)  # samples*features
        y = label_list
        clf = svm.SVC(probability=True)
        clf.fit(X, y)
        self.model = clf

    def predict(self, model, board_shim, board_id, true_label):
        sample_to_predict = np.array(collect_features(board_shim, board_id))
        result = model.predict([sample_to_predict])
        prob = model.predict_proba([sample_to_predict])
        print('prediction result:', result, 'probability:', prob, 'true label:', true_label)


def collect_features(board_shim, board_id):
    collected_data = []
    max_sample_num = 1000  # max samples used for feature extracting
    previous_timestamp = 0
    chan_timestamp = board_shim.get_timestamp_channel(board_id)

    while len(collected_data) < max_sample_num:
        data = board_shim.get_current_board_data(256).T
        batch_size = data.shape[0]
        for i in range(batch_size):
            sample = data[i, :].tolist()
            current_timestamp = sample[chan_timestamp]
            if current_timestamp > previous_timestamp:
                previous_timestamp = current_timestamp
                collected_data.append(sample)
            else:
                continue
    n_win = 20
    win_len = int(max_sample_num/n_win)
    chan_for_feature = [1, 2, 3, 4]
    collected_data = np.array(collected_data)[:max_sample_num, chan_for_feature]  # select channels for feature extracting
    collected_data = np.concatenate(collected_data.T)
    feature_vec = [np.mean(collected_data[k: k+win_len]) for k in range(0, len(collected_data), win_len)]

    return feature_vec


def thread_event(board_shim, board_id):
    streams = resolve_byprop("name", "SendMarkersOnClick")
    inlet = StreamInlet(streams[0])
    feature_list = []
    label_list = []
    dict_clf = {}

    while True:
        event_sample, event_timestamp = inlet.pull_sample()
        print("got %s at time %s" % (event_sample[0], event_timestamp))
        message_list = event_sample[0].split(';')
        if message_list[0] == 'collect':
            feature_vec = collect_features(board_shim, board_id)
            feature_list.append(feature_vec)
            label_list.append(int(message_list[2]))
        if message_list[0] == 'train':
            if len(label_list) < 5:
                print('too few samples, please collect more!')
            else:
                cf = ClassifierFactory(message_list[2])
                classifier = cf(feature_list, label_list)
                classifier.train(feature_list, label_list)
                dict_clf[message_list[1]] = classifier
                for key, value in dict_clf.items():
                    print(key, value, vars(value))
        if message_list[0] == 'predict':
            if message_list[1] not in dict_clf:
                print('no classifier named', message_list[1], 'found, please train classifier first!')
            else:
                clf = dict_clf[message_list[1]]
                clf.predict(clf.model, board_shim, board_id, message_list[2])


def start_all(board_id, params, streamparams, calib_length, power_length, scale, offset, head_impact):
    board_shim = BoardShim(board_id, params)
    board_shim.prepare_session()
    board_shim.config_board("p61")
    board_shim.start_stream(450000, streamparams)

    thread1 = threading.Thread(target=Graph, args=(board_shim, calib_length, power_length, scale, offset, head_impact), daemon=True)
    thread1.start()
    thread2 = threading.Thread(target=thread_event, args=(board_shim, board_id), daemon=True)
    thread2.start()

    print(board_shim.get_board_descr(board_id))
    n_chan = board_shim.get_num_rows(board_id)
    rate = board_shim.get_sampling_rate(board_id)
    info_data = StreamInfo('Z-flow-data', 'EEG', n_chan, rate, 'float32', 'zflow_SendData')
    channel_names = ["packagenum","TP9","Fp1","Fp2","TP10","accel1","accel2","accel3","gyro1","gyro2","gyro3","ppg1","ppg2","ppg3","timestamp","marker"]
    info_data.desc().append_child_value("manufacturer", "Brainflow")
    chns = info_data.desc().append_child("channels")
    for chan_ix, label in enumerate(channel_names):
        ch = chns.append_child("channel")
        ch.append_child_value("label", label)
    outlet_data = StreamOutlet(info_data)
    chan_timestamp = board_shim.get_timestamp_channel(board_id)

    previous_timestamp = 0
    while True:
        data = board_shim.get_current_board_data(256).T
        batch_size = data.shape[0]
        for i in range(batch_size):
            sample = data[i, :].tolist()
            current_timestamp = sample[chan_timestamp]
            if current_timestamp > previous_timestamp:
                previous_timestamp = current_timestamp
                outlet_data.push_sample(sample, sample[chan_timestamp])
            else:
                continue
        time.sleep(0.01)


def main():
    BoardShim.enable_dev_board_logger()
    # BoardShim.set_log_file('test_board.log')
    # logging.basicConfig(filename='example.log', level=logging.DEBUG)
    logging.basicConfig(filename='example.log', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=30)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.MUSE_S_BOARD)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    calib_length = 600
    power_length = 10
    scale = 1.5
    offset = 0.5
    head_impact = 0.2

    board_id = args.board_id
    streamparams = args.streamer_params

    start_all(board_id, params, streamparams, calib_length, power_length, scale, offset, head_impact)


def connect(board_id, timeout, calib_length, power_length, scale, offset, head_impact, record):
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)
    params = BrainFlowInputParams()
    params.timeout = timeout

    if record == 1:
        streamparams = "file://braindata.tsv:w"
    else:
        streamparams = ""

    start_all(board_id, params, streamparams, calib_length, power_length, scale, offset, head_impact)



if __name__ == '__main__':
    main()
