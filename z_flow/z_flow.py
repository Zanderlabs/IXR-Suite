import argparse
import logging
from threading import Event

from brainflow.board_shim import (BoardIds, BoardShim, BrainFlowInputParams)

from z_flow.render import Graph
from z_flow.lsl_utility import LslEventListener, BfLslDataPublisher


class ZFlow:
    def __init__(self) -> None:
        self.board_shim = None
        self.board_id = None

        self.calib_length = 600
        self.power_length = 10
        self.scale = 1.5
        self.offset = 0.5
        self.head_impact = 0.2

        self.classifiers = {}

    def __del__(self):
        if isinstance(self.board_shim, BoardShim):
            self.board_shim.stop_stream()
            self.board_shim.release_session()

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

        self.module_launcher()

    def module_launcher(self) -> None:
        """Method that starts and execute the three main functions of z-flow, in their own thread of control.
        """
        stay_alive = Event()
        stay_alive.set()

        graph_thread = Graph(self.board_shim, thread_name="graph_1", thread_daemon=False)
        lsl_event_listener = LslEventListener(self.board_shim, stay_alive=stay_alive,
                                              thread_name="lsl_event_listener_1", thread_daemon=False)
        lsl_data_pusher = BfLslDataPublisher(self.board_shim, stay_alive=stay_alive,
                                        thread_name="lsl_data_pusher_1", thread_daemon=False)

        graph_thread.start()
        lsl_event_listener.start()
        lsl_data_pusher.start()

        # Runs as long as z-flow bashboard is not terminated by user.
        graph_thread.join()

        # then shutdown remaining threads gracefully.
        stay_alive.clear()
        lsl_event_listener.join()
        lsl_data_pusher.join()

        print("Closing down z-flow")

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

        self.module_launcher()

    def _create_board_shim(self, board_id: BoardIds, params: BrainFlowInputParams, streamparams: str) -> None:
        self.board_id = BoardIds(board_id)
        self.board_shim = BoardShim(board_id, params)
        self.board_shim.prepare_session()
        self.board_shim.config_board("p61")
        self.board_shim.start_stream(450000, streamparams)
