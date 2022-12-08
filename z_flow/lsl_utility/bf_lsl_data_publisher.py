import logging
import time
from threading import Event, Thread

from brainflow import BoardShim, BrainFlowPresets, BrainFlowError
from pylsl import StreamInfo, StreamOutlet, local_clock, cf_double64


class BfLslDataPublisher(Thread):
    """Class that pulls data (eeg, ppg, gyro) from Brainflow and
    pushes it over LSL as raw data, executed in it's own thread of control.

    The instance will automatically shutdown if the stay_alive event has been cleared.

    Extends from threading.Thread, for more information:
    https://docs.python.org/3/library/threading.html#thread-objects

    If thread_daemon the parameter is set, the thread is launched in daemon mode,
    the significance of this flag is that a deamon thread does not keep the Python process alive.

    :param board_shim: Brainflow BoardShim to collect data from EEG devices.
    :type board_shim: BoardShim
    :param stay_alive: Life line to indicate that the thread should stay alive.
    :type stay_alive: Event
    :param thread_name: Thread name, defaults to "lsl_data_pusher"
    :type thread_name: str, optional
    :param thread_daemon: Sets thread as daemon, or not, defaults to False
    :type thread_daemon: bool, optional
    """

    def __init__(self, board_shim: BoardShim, stay_alive: Event,
                 thread_name: str = "lsl_data_pusher", thread_daemon: bool = False) -> None:
        Thread.__init__(self, name=thread_name, daemon=thread_daemon)
        self.stay_alive = stay_alive
        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        self.data_types = {
            'eeg': BrainFlowPresets.DEFAULT_PRESET,
            'gyro': BrainFlowPresets.AUXILIARY_PRESET,
            'ppg': BrainFlowPresets.ANCILLARY_PRESET,
        }
        self.channels = {k: self.get_channels(v) for k, v in self.data_types.items()}
        self.outlets = {}
        self.previous_timestamp = {'eeg': 0, 'gyro': 0, 'ppg': 0}
        self.local2lsl_time_diff = time.time() - local_clock()  # compute time difference with LSL system.

    def run(self) -> None:
        """Once a thread object is created, its activity must be started by calling the thread’s start() method.
        This invokes the run() method in a separate thread of control.
        """
        for data_type, preset in self.data_types.items():
            rate = self.board_shim.get_sampling_rate(self.board_id, preset)
            name = f'z-flow-{data_type}-data'

            logging.info(f"Starting '{name}' LSL Data Publisher stream.")
            info_data = StreamInfo(name=name, type=data_type, channel_count=len(self.channels[data_type]), nominal_srate=rate,
                                   channel_format=cf_double64, source_id='z-flow-lsl-data-publisher')
            stream_channels = info_data.desc().append_child("channels")
            for _, label in self.channels[data_type].items():
                ch = stream_channels.append_child("channel")
                ch.append_child_value("label", label)
                if data_type == 'eeg':
                    ch.append_child_value("unit", 'microvolts')
                ch.append_child_value("type", data_type)
            self.outlets[data_type] = StreamOutlet(info_data)
            logging.info(f"'{self.outlets[data_type].get_info().name()}' LSL Data Publisher stream started.")

        while self.stay_alive.is_set():
            if not self.board_shim.is_prepared():
                # if no connection is established, try again later.
                continue

            for data_type, preset in self.data_types.items():
                timestamp_column = self.board_shim.get_timestamp_channel(self.board_id, preset=preset)

                try:
                    data = self.board_shim.get_current_board_data(1024, preset)
                except BrainFlowError as e:
                    # Right after board preparation the Brainflow connection might be a bit unstable.
                    # In that case Brainflow throws an INVALID_ARGUMENTS_ERROR exception.
                    # If that case, try again later, but re-raise other exceptions.
                    if e.exit_code == BrainFlowError.INVALID_ARGUMENTS_ERROR:
                        continue
                    else:
                        raise e

                # slice rows with timestamps bigger then previous_timestamp
                data = data[:, data[timestamp_column] > self.previous_timestamp[data_type]]

                # only update timestamp and push if there is something left to push.
                if data.shape[1] > 0:
                    self.previous_timestamp[data_type] = data[timestamp_column, -1]
                    data = data[list(self.channels[data_type].keys()), :]
                    self.outlets[data_type].push_chunk(data.T.tolist(),
                                                       self.previous_timestamp[data_type] - self.local2lsl_time_diff)
            time.sleep(1)

    def get_channels(self, preset: BrainFlowPresets) -> dict[int, str]:
        channels = {}
        description = BoardShim.get_board_descr(self.board_id, preset)
        if preset == BrainFlowPresets.DEFAULT_PRESET:
            channels.update(dict(zip(description['eeg_channels'], description['eeg_names'].split(","))))
        elif preset == BrainFlowPresets.AUXILIARY_PRESET:
            channels.update({channel: f"accel_{i}" for i, channel in enumerate(description['accel_channels'])})
            channels.update({channel: f"gyro_{i}" for i, channel in enumerate(description['gyro_channels'])})
        elif preset == BrainFlowPresets.ANCILLARY_PRESET:
            channels.update({channel: f"ppg_{i}" for i, channel in enumerate(description['ppg_channels'])})
        else:
            raise ValueError("Unrecognized BrainFlowPresets")
        return channels
