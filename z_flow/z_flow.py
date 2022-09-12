import argparse
import logging
from pathlib import Path
from threading import Event
from time import strftime

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams, BrainFlowError

from z_flow.lsl_utility import BfLslDataPublisher, LslEventListener
from z_flow.render import Graph


class ZFlow:
    def __init__(self, args: list[str] | None = None) -> None:
        self.board_shim = None
        self.classifiers = {}

        parser = self.create_parser()
        self.args = parser.parse_args(args)

        log_file_path = Path(self.args.log_file)
        if log_file_path.exists():
            logging.getLogger().warning(f"A file with this name {self.args.log_file} already exists. "
                                        "The old file has been renamed with a timestamp.")
            log_file_path.rename(log_file_path.with_stem(f'{log_file_path.stem}_{strftime("%Y-%m-%d_%H-%M-%S")}'))

        if self.args.log_brainflow:
            BoardShim.enable_board_logger()
            BoardShim.set_log_file(self.args.log_file)

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(threadName)s] [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.args.log_file),
                logging.StreamHandler()  # default is to stream to sys.stdout
            ]
        )

    def __del__(self):
        logging.info("Closing down Brainflow session.")
        if isinstance(self.board_shim, BoardShim):
            self.board_shim.stop_stream()
            self.board_shim.release_session()

    def run(self) -> None:
        params = BrainFlowInputParams()
        params.timeout = self.args.timeout
        params.mac_address = self.args.mac_address
        params.serial_number = self.args.serial_number

        logging.info("Starting brainflow session.")
        self.board_shim = BoardShim(self.args.board_id, params)
        self.board_shim.prepare_session()
        logging.info("Board connected, configuring.")
        self.board_shim.config_board("p61")
        logging.info("Board configured, starting stream.")
        self.board_shim.start_stream(450000, self.args.streamer_params)

        self.module_launcher()

    def module_launcher(self) -> None:
        """Method that starts and execute the three main functions of z-flow, in their own thread of control.
        """
        stay_alive = Event()
        stay_alive.set()

        logging.info("Starting dashboard.")
        graph_thread = Graph(self.board_shim, thread_name="graph_1", thread_daemon=False)
        graph_thread.set_parameters(self.args.calib_length, self.args.power_length,
                                    self.args.scale, self.args.offset, self.args.head_impact)
        graph_thread.start()

        logging.info("Starting LSL event listener.")
        lsl_event_listener = LslEventListener(self.board_shim, stay_alive=stay_alive, thread_daemon=False)
        lsl_event_listener.start()

        logging.info("Starting Brainflow LSL data publisher.")
        lsl_data_pusher = BfLslDataPublisher(self.board_shim, stay_alive=stay_alive, thread_daemon=False)
        lsl_data_pusher.start()

        logging.info("Running Z-flow as long as the dashboard is open, please close the dashboard to close Z-flow.")
        graph_thread.join()

        logging.info("Z-flow dashboard closed, terminating all child threads.")
        stay_alive.clear()
        lsl_event_listener.join()
        lsl_data_pusher.join()
        logging.info("Main threads closed down.")

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Z-flow")

        # Brainflow arguments
        # use docs to check which parameters are required for specific board, for muse products, see specifically:
        # https://brainflow.readthedocs.io/en/stable/SupportedBoards.html#muse
        parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                            required=False, default=BoardIds.MUSE_S_BOARD)
        parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                            default=30)
        parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
        parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
        parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')

        # Z-flow Dashboard arguments
        parser.add_argument('--calib-length', type=int, default=600, help='Calibration length, defaults to 600')
        parser.add_argument('--power-length', type=int, default=10, help='Power length, defaults to 10')
        parser.add_argument('--scale', type=float, default=1.5, help='Scale, defaults to 1.5')
        parser.add_argument('--offset', type=float, default=0.5, help='Offset, defaults to 0.5')
        parser.add_argument('--head-impact', type=float, default=0.2, help='Head impact, defaults to 0.2')

        # Z-flow utility arguments
        parser.add_argument('--log-file', type=str, default='z_flow.log', required=False,
                            help="The file name where the Z-flow will write it's log's to. "
                                 "If a relative path is given it is relative to the current working directory.")
        parser.add_argument('--log-brainflow', type=bool, default=False, required=False,
                            help="Also write Brainflow logs to log file.")
        return parser
