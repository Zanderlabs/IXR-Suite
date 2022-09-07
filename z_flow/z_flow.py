import argparse
import logging
from threading import Event

from brainflow.board_shim import (BoardIds, BoardShim, BrainFlowInputParams)

from z_flow.render import Graph
from z_flow.lsl_utility import LslEventListener, BfLslDataPublisher


class ZFlow:
    def __init__(self, args: list[str] | None = None) -> None:
        self.board_shim = None
        self.classifiers = {}

        parser = self.create_parser()
        self.args = parser.parse_args(args)

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

        params = BrainFlowInputParams()
        params.timeout = self.args.timeout
        params.mac_address = self.args.mac_address
        params.serial_number = self.args.serial_number

        self.board_shim = BoardShim(self.args.board_id, params)
        self.board_shim.prepare_session()
        self.board_shim.config_board("p61")
        self.board_shim.start_stream(450000, self.args.streamer_params)

        self.module_launcher()

    def module_launcher(self) -> None:
        """Method that starts and execute the three main functions of z-flow, in their own thread of control.
        """
        stay_alive = Event()
        stay_alive.set()

        graph_thread = Graph(self.board_shim, thread_name="graph_1", thread_daemon=False)
        graph_thread.set_parameters(self.args.calib_length, self.args.power_length,
                                    self.args.scale, self.args.offset, self.args.head_impact)
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

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Z-flow")
        # use docs to check which parameters are required for specific board, for muse products, see specifically:
        # https://brainflow.readthedocs.io/en/stable/SupportedBoards.html#muse
        parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                            required=False, default=BoardIds.MUSE_S_BOARD)
        parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                            default=30)
        parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
        parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
        parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
        parser.add_argument('--calib-length', type=int, default=600, help='Calibration length, defaults to 600')
        parser.add_argument('--power-length', type=int, default=10, help='Power length, defaults to 10')
        parser.add_argument('--scale', type=float, default=1.5, help='Scale, defaults to 1.5')
        parser.add_argument('--offset', type=float, default=0.5, help='Offset, defaults to 0.5')
        parser.add_argument('--head-impact', type=float, default=0.2, help='Head impact, defaults to 0.2')
        return parser
