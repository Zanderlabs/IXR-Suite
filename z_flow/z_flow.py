import argparse
import logging
import threading
import time

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams, BrainFlowPresets
from brainflow.exit_codes import *
from pylsl import (StreamInfo, StreamInlet, StreamOutlet, local_clock,
                   resolve_byprop)

from z_flow.render import Graph
from z_flow.classifiers import Classifier, LDA, SVM


class ZFlow:
    def __init__(self) -> None:
        self.board_shim = None
        self.board_id = None

        self.calib_length = 600
        self.power_length = 10
        self.scale = 1.5
        self.offset = 0.5
        self.head_impact = 0.2

    def run(self) -> None:
        BoardShim.enable_dev_board_logger()
        # BoardShim.set_log_file('test_board.log')
        # logging.basicConfig(filename='example.log', level=logging.DEBUG)
        logging.basicConfig(filename='example.log',
                            format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG)
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

        self._create_board_shim(args.board_id, params, args.streamer_params)

        self.start_all()

    def message_decode(self, message: str, event_timestamp: float, dict_clf: dict) -> None:
        message_list = message.split(';')
        if message_list[0] == 'create':
            cf = ClassifierFactory(message_list[2])
            classifier = cf(message_list[2], message_list[3], message_list[4], message_list[5])
            dict_clf[message_list[1]] = classifier

        if message_list[0] == 'collect':
            clf_name = message_list[1]
            if clf_name not in dict_clf:
                print('no classifier named', clf_name, 'found, please create classifier first!')
            else:
                clf = dict_clf[clf_name]
                clf.feature_list.append(clf.collect_sample(self.board_shim, self.board_id, event_timestamp))
                clf.label_list.append(int(message_list[2]))
                print(np.array(clf.feature_list).shape, clf.label_list)

        if message_list[0] == 'train':
            clf_name = message_list[1]
            if clf_name not in dict_clf:
                print('no classifier named', clf_name, 'found, please create classifier first!')
            else:
                clf = dict_clf[clf_name]
                clf.train()

        if message_list[0] == 'predict':
            clf_name = message_list[1]
            if clf_name not in dict_clf:
                print('no classifier named', clf_name, 'found, please create classifier first!')
            else:
                clf = dict_clf[clf_name]
                clf.predict(self.board_shim, self.board_id, event_timestamp)

        if message_list[0] == 'dictionary':
            str_dict_info = ''
            for key, value in dict_clf.items():
                print(vars(value))
            #     str_clf_info = ''
            #     for k, v in vars(value).items():
            #         if not isinstance(v, str):
            #             if isinstance(v, dict):
            #                 v = json.dumps(v)
            #             else:
            #                 v = " ".join(map(str, v))
            #         str_clf_info = str_clf_info + k + ': ' + v + ', '
            #     str_dict_info = str_dict_info + key + ': ' + str_clf_info + ' ###### '
            # outlet.push_sample([str_dict_info])

    def thread_event(self) -> None:
        dict_clf = {}
        # create LSL stream to receive events
        streams = resolve_byprop("name", "SendMarkersOnClick")
        inlet = StreamInlet(streams[0])
        # create LSL stream to send information printed on console
        info = StreamInfo('SendPrintInfo', 'Markers', 1, 0, 'string', 'Zflow_SendPrintInfo')
        outlet = StreamOutlet(info)

        while True:
            event_sample, event_timestamp = inlet.pull_sample()
            diff = time.time() - local_clock()
            event_timestamp = event_timestamp+diff  # get timestamp with Unix Epoch format
            print("got %s at time %s" % (event_sample[0], event_timestamp))
            outlet.push_sample(["got %s at time %s" % (event_sample[0], event_timestamp)])
            self.message_decode(event_sample[0], event_timestamp, dict_clf)

    def start_all(self) -> None:
        thread1 = threading.Thread(target=Graph, args=(self.board_shim, self.calib_length,
                                   self.power_length, self.scale, self.offset, self.head_impact), daemon=True)
        thread1.start()
        thread2 = threading.Thread(target=self.thread_event, daemon=True)
        thread2.start()
        # asyncio.run(thread_event(board_shim, board_id))

        print(self.board_shim.get_board_descr(self.board_id))

        data_types = {
            'eeg': BrainFlowPresets.DEFAULT_PRESET,
            'gyro': BrainFlowPresets.AUXILIARY_PRESET,
            'ppg': BrainFlowPresets.ANCILLARY_PRESET,
        }
        outlets = {}

        for data_type, preset in data_types.items():
            n_chan = self.board_shim.get_num_rows(self.board_id, preset)
            rate = self.board_shim.get_sampling_rate(self.board_id, preset)
            info_data = StreamInfo('Z-flow-data', data_type, n_chan, rate, 'float32', 'zflow_SendData')
            info_data.desc().append_child_value("manufacturer", "Brainflow")
            info_data.desc().append_child_value("description", str(
                BoardShim.get_board_descr(self.board_id, preset)))
            outlets[data_type] = StreamOutlet(info_data)

        previous_timestamp = {'eeg': 0, 'gyro': 0, 'ppg': 0}
        diff = time.time() - local_clock()

        while True:
            for data_type, preset in data_types.items():
                timestamp_column = self.board_shim.get_timestamp_channel(self.board_id, preset=preset)

                # TODO: Potential cause for a bug here in the following lines
                # if (loop/thread) latency is longer then 1024/sample_rate we skip frames of data. This should be resolved.
                data = self.board_shim.get_current_board_data(1024, preset)
                # slice rows with timestamps bigger then previous_timestamp
                data = data[:, data[timestamp_column] > previous_timestamp[data_type]]

                # only update timestamp and push if there is something left to push.
                if data.shape[1] > 0:
                    previous_timestamp[data_type] = data[timestamp_column, -1]
                    outlets[data_type].push_chunk(data.tolist(), previous_timestamp[data_type]-diff)

    def connect(self, board_id: BoardIds, timeout: int, calib_length: int, power_length: int,
                scale: float, offset: float, head_impact: float, record) -> None:
        self.calib_length = calib_length
        self.power_length = power_length
        self.scale = scale
        self.offset = offset
        self.head_impact = head_impact

        BoardShim.enable_dev_board_logger()
        logging.basicConfig(level=logging.DEBUG)
        params = BrainFlowInputParams()
        params.timeout = timeout

        if record == 1:
            streamparams = "file://braindata.tsv:w"
        else:
            streamparams = ""

        self._create_board_shim(board_id, params, streamparams)

        self.start_all()

    def _create_board_shim(self, board_id: BoardIds, params: BrainFlowInputParams, streamparams: str) -> None:
        self.board_id = BoardIds(board_id)
        self.board_shim = BoardShim(board_id, params)
        self.board_shim.prepare_session()
        self.board_shim.config_board("p61")
        self.board_shim.start_stream(450000, streamparams)


class ClassifierFactory:
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type

    def __call__(self, *args, **kwargs) -> any:
        if self.model_type == 'LDA':
            return LDA(*args, **kwargs)
        elif self.model_type == 'SVM':
            return SVM(*args, **kwargs)
        else:
            raise ValueError("Type not known!")
