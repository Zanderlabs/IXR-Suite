import argparse
import logging
from pathlib import Path
from threading import Event
from time import strftime

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from z_flow.board import BrainFlowHandler
from z_flow.lsl_utility import BfLslDataPublisher, LslEventListener, LslLogger
from z_flow.gui import ZDashboard


class ZFlow:
    def __init__(self, args: list[str] | None = None) -> None:
        self.board_shim = None
        self.classifiers = {}

        parser = self.create_parser()
        self.args = parser.parse_args(args)

        log_file_path = Path(self.args.log_file)
        if not log_file_path.parent.exists():
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
        if log_file_path.exists():
            logging.getLogger().warning(f"A file with this name {self.args.log_file} already exists. "
                                        "The old file has been renamed with a timestamp.")
            log_file_path.rename(log_file_path.with_stem(f'{log_file_path.stem}_{strftime("%Y-%m-%d_%H-%M-%S")}'))

        if self.args.log_brainflow:
            BoardShim.enable_board_logger()
            BoardShim.set_log_file(self.args.log_file)

        # configure root logger
        handlers = [
            logging.StreamHandler(),  # sys.stdout handler
            logging.FileHandler(self.args.log_file),
        ]
        if self.args.lsl_log:
            handlers.append(LslLogger('z-flow-log'))
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(threadName)s] [%(levelname)s] %(message)s",
            handlers=handlers
        )

    def __del__(self) -> None:
        logging.getLogger().handlers = []  # release all root logger handlers

    def run(self) -> None:
        params = BrainFlowInputParams()
        params.timeout = self.args.timeout

        stay_alive = Event()
        stay_alive.set()

        logging.info("Starting Brainflow Session (with Bluetooth connection)")
        board_shim = BoardShim(self.args.board_id, params)
        brainflow_thread = BrainFlowHandler(board_shim, params, stay_alive, self.args.streamer_params)
        brainflow_thread.start()

        logging.info("Starting dashboard.")
        dashboard_thread = ZDashboard(board_shim, self.args.reference, self.args.display_ref,
                                      thread_name="graph_1", thread_daemon=False)
        dashboard_thread.set_parameters(self.args.calib_length, self.args.power_length,
                                        self.args.scale, self.args.offset, self.args.head_impact)
        dashboard_thread.start()

        logging.info("Starting LSL event listener.")
        lsl_event_listener_thread = LslEventListener(board_shim, reference=self.args.reference,
                                                     stay_alive=stay_alive, thread_daemon=False)
        lsl_event_listener_thread.start()

        logging.info("Starting Brainflow LSL data publisher.")
        lsl_data_pusher_thread = BfLslDataPublisher(board_shim, push_full_vec=self.args.push_full_vec, stay_alive=stay_alive, thread_daemon=False)
        lsl_data_pusher_thread.start()

        logging.info("Running Z-flow as long as the dashboard is open, please close the dashboard to close Z-flow.")
        dashboard_thread.join()

        logging.info("Z-flow dashboard closed, terminating all child threads.")
        stay_alive.clear()
        lsl_event_listener_thread.join()
        lsl_data_pusher_thread.join()
        brainflow_thread.join()
        brainflow_thread.release_brainflow()
        logging.info("Successfully shutdown.")

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
        parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')

        # re-referencing options.
        parser.add_argument('--reference', type=str, default='mean', choices=['none', 'mean', 'ref'],
                            help="Determines what type of re-reference to use. "
                                 " - none: No re-referencing is applied."
                                 " - mean (default): Use the mean of the four frontal and temporal electrodes."
                                 " - ref: Use the reference electrode(s) as a reference.")
        parser.add_argument('--display-ref', action='store_true',
                            help="Displays signal of the reference electrode(s) on the dashboard. ")

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
        parser.add_argument('--no-lsl-log', action='store_false', dest='lsl_log',
                            help="Disables logging over lsl.")
        parser.add_argument('--push_full_vec', action='store_true',
                            help='Push the full vector over LSL received by Brainflow.')
        return parser
