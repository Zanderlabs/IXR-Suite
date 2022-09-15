import logging
from threading import Thread

import numpy as np
import pyqtgraph as pg
from brainflow import (BoardShim, BrainFlowPresets, DataFilter,
                       DetrendOperations, FilterTypes, WindowOperations)
from pylsl import StreamInfo, StreamOutlet
from pyqtgraph.Qt import QtCore, QtGui


class Graph(Thread):
    """Class that implements a basic dashboard to
    display EEG, PPG, motion, brain waves, and z-flow metrics.

    It also pushes computed power metrics over LSL.

    Extends from threading.Thread, for more information:
    https://docs.python.org/3/library/threading.html#thread-objects

    If thread_daemon parameter sets the thread to daemon mode,
    the significance of this flag is that the entire Python program
    exits when only daemon threads are left.

    :param board_shim: Brainflow BoardShim to collect data from EEG devices.
    :type board_shim: BoardShim
    :param thread_name: Thread name, defaults to "graph"
    :type thread_name: str, optional
    :param thread_daemon: Sets thread as daemon, or not, defaults to False
    :type thread_daemon: bool, optional
    """

    def __init__(self, board_shim: BoardShim, thread_name: str = "thread_graph", thread_daemon: bool = False) -> None:
        Thread.__init__(self, name=thread_name, daemon=thread_daemon)
        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()

        pg.setConfigOption('background', '#264653')
        pg.setConfigOption('foreground', '#e9f5db')

        self.eeg_preset = BrainFlowPresets.DEFAULT_PRESET
        self.gyro_preset = BrainFlowPresets.AUXILIARY_PRESET
        self.ppg_preset = BrainFlowPresets.ANCILLARY_PRESET

        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id, self.eeg_preset)
        self.gyro_channels = BoardShim.get_gyro_channels(self.board_id, self.gyro_preset)
        self.ppg_channels = BoardShim.get_ppg_channels(self.board_id, self.ppg_preset)
        self.eeg_sampling_rate = BoardShim.get_sampling_rate(self.board_id, self.eeg_preset)
        self.gyro_sampling_rate = BoardShim.get_sampling_rate(self.board_id, self.gyro_preset)
        self.ppg_sampling_rate = BoardShim.get_sampling_rate(self.board_id, self.ppg_preset)
        self.update_speed_ms = 100
        self.plot_window_s = 10  # should always be bigger then power_metric_window_ms
        self.power_metric_window_s = 1.5  # should always be bigger then psd size
        self.psd_size = DataFilter.get_nearest_power_of_two(self.eeg_sampling_rate)

        # selfmade power metrics
        self.set_parameters()

        self.inverse_workload_calib = [0, 1]
        self.inverse_workload_hist = [0, 1]
        self.inverse_workload = 0
        self.engagement_calib = [0, 1]
        self.engagement_hist = [0, 1]
        self.engagement = 0
        self.power_metrics = 0

        # LSL stream
        name = 'BrainPower'
        logging.info(f"Starting '{name}' Power Metric stream.")
        info_transmit = StreamInfo(name, 'Z-metric', 1, 0, 'float32', 'zflow_transmit_power')
        self.outlet_transmit = StreamOutlet(info_transmit)
        logging.info(f"'{self.outlet_transmit.get_info().name()}' Power Metric stream started.")

    def run(self):
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='Z-flow', size=(1500, 1000))

        self._init_pens()
        self._init_timeseries()
        self._init_psd()
        self._init_band_plot()
        self._init_brain_power_plot()

        timer = QtCore.QTimer()
        timer.timeout.connect(self._update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def set_parameters(self, calib_length: int = 600, power_length: int = 10, scale: float = 1.5,
                       offset: float = 0.5, head_impact: float = 0.2) -> None:
        """Allows setting z-flow metrics. Is called with defaults on object initialization.

        :param calib_length: Calibration length, defaults to 600
        :type calib_length: int, optional
        :param power_length: Power length, defaults to 10
        :type power_length: int, optional
        :param scale: Scale, defaults to 1.5
        :type scale: float, optional
        :param offset: Offset, defaults to 0.5
        :type offset: float, optional
        :param head_impact: Head impact, defaults to 0.2
        :type head_impact: float, optional
        """
        self.calib_length = int(calib_length * 1000 / self.update_speed_ms)
        self.hist_length = int(power_length * 1000 / self.update_speed_ms)
        self.brain_scale = scale
        self.brain_center = offset
        self.head_impact = head_impact

    def _init_pens(self) -> None:
        self.pens = list()
        self.brushes = list()
        colors = ['#e9c46a', '#f4a261', '#e76f51', '#d62828', '#2a9d8f', '#168aad', '#e9f5db', '#A57E2F', '#A53B2F']
        for i in range(len(colors)):
            pen = pg.mkPen({'color': colors[i], 'width': 2})
            self.pens.append(pen)
            brush = pg.mkBrush(colors[i])
            self.brushes.append(brush)

    def _init_timeseries(self) -> None:
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

    def _init_psd(self) -> None:
        self.psd_plot = self.win.addPlot(row=0, col=1, rowspan=4)
        self.psd_plot.showAxis('left', False)
        self.psd_plot.setMenuEnabled('left', False)
        self.psd_plot.setTitle('spectral power')
        self.psd_plot.setLogMode(False, True)
        self.psd_plot.setLabel('bottom', 'frequency (Hz)')
        self.psd_plot.setXRange(0, 50, padding=0)
        self.psd_curves = list()
        for i in range(len(self.eeg_channels)):
            psd_curve = self.psd_plot.plot(pen=self.pens[i % len(self.pens)])
            psd_curve.setDownsampling(auto=True, method='mean', ds=3)
            self.psd_curves.append(psd_curve)

    def _init_band_plot(self) -> None:
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

    def _init_brain_power_plot(self) -> None:
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

    def _update(self) -> None:
        eeg_data = self.board_shim.get_current_board_data(self.plot_window_s * self.eeg_sampling_rate,
                                                          self.eeg_preset)[self.eeg_channels, :]
        gyro_data = self.board_shim.get_current_board_data(self.plot_window_s * self.gyro_sampling_rate,
                                                           self.gyro_preset)[self.gyro_channels, :]
        # Only pick the first of the PPG channels, which is channel 1 (zero indexed) of the board data array
        ppg_data = self.board_shim.get_current_board_data(self.plot_window_s * self.ppg_sampling_rate,
                                                          self.ppg_preset)[self.ppg_channels[0], :]

        # mean normalize
        eeg_data = eeg_data - np.mean(eeg_data, axis=0)

        # add gyro data to curves, leave first few curves for eeg data.
        for count, _ in enumerate(self.gyro_channels):
            self.curves[len(self.eeg_channels) + count].setData(gyro_data[count].tolist())
        head_movement = np.clip(np.mean(np.abs(gyro_data)) / 50, 0, 1)
        #  power_metrics[2] = head_movement

        # ppg: filter and add ppg to curves, again at the appropriate index.
        DataFilter.detrend(ppg_data, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(data=ppg_data, sampling_rate=self.ppg_sampling_rate, start_freq=0.8,
                                    stop_freq=4.0, order=4, filter_type=FilterTypes.BUTTERWORTH.value, ripple=0.0)
        self.curves[eeg_data.shape[0] + gyro_data.shape[0]].setData(ppg_data.tolist())

        # eeg processing
        avg_bands = [0, 0, 0, 0, 0]
        frontal_theta = 1
        parietal_alpha = 1
        engagement_idx = 1

        for count, _ in enumerate(self.eeg_channels):
            DataFilter.detrend(eeg_data[count], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data=eeg_data[count], sampling_rate=self.eeg_sampling_rate, start_freq=1.0,
                                        stop_freq=59.0, order=2, filter_type=FilterTypes.BUTTERWORTH.value, ripple=0.0)
            DataFilter.perform_bandstop(data=eeg_data[count], sampling_rate=self.eeg_sampling_rate, start_freq=48.0,
                                        stop_freq=52.0, order=2, filter_type=FilterTypes.BUTTERWORTH.value, ripple=0.0)
            # plot timeseries
            self.curves[count].setData(eeg_data[count].tolist())

            # take/slice the last samples of eeg_data that fall within the power metric window
            eeg_data_pm_sliced = eeg_data[count][-(self.power_metric_window_s * self.eeg_sampling_rate):]
            if len(eeg_data_pm_sliced) < self.psd_size:
                continue  # First time _update() runs there is not enough data yet to compute psd

            # compute psd
            psd_data = DataFilter.get_psd_welch(data=eeg_data_pm_sliced,
                                                nfft=self.psd_size,
                                                overlap=self.psd_size // 2,
                                                sampling_rate=self.eeg_sampling_rate,
                                                window=WindowOperations.BLACKMAN_HARRIS.value)
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

        # weighted mean
        engagement_weighted_mean = 0
        sumweight = 0
        for count, hist_val in enumerate(self.engagement_hist):
            engagement_weighted_mean += hist_val * count
            sumweight += count

        engagement_weighted_mean = engagement_weighted_mean / sumweight

        self.engagement = engagement_weighted_mean
        self.power_metrics = np.float32(self.engagement + (1 - head_movement) * self.head_impact)

        # plot bars
        self.band_bar.setOpts(height=avg_bands)
        self.power_bar.setOpts(height=self.power_metrics)

        self.outlet_transmit.push_sample([self.power_metrics])

        self.app.processEvents()
