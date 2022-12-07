import logging
from threading import Event, Thread
from time import time, sleep

from brainflow import BoardIds, BoardShim, BrainFlowError, BrainFlowInputParams, BrainFlowPresets, BrainFlowExitCodes


class BrainFlowHandler(Thread):
    def __init__(self,
                 board_shim: BoardShim,
                 params: BrainFlowInputParams,
                 stay_alive: Event,
                 streamer_params: str | None = None,
                 thread_name: str = "thread_brainflow",
                 thread_daemon: bool = False) -> None:
        Thread.__init__(self, name=thread_name, daemon=thread_daemon)
        self.board_shim = board_shim
        self.board_id = self.board_shim.get_board_id()
        self.streamer_params = streamer_params
        self.stay_alive = stay_alive
        self.time_out = params.timeout
        self.ringbuffer_size = 45_000

    def run(self) -> None:
        while self.stay_alive.is_set():
            if not self.board_shim.is_prepared():
                logging.info("Starting brainflow session.")
                try:
                    self._prepare_board()
                    logging.info("Succesfully started brainflow session.")
                    # Brainflow and the Muse S need a few seconds after board preparation to actually start sending data.
                    sleep(2)
                except BrainFlowError as e:
                    if e.exit_code == BrainFlowExitCodes.BOARD_NOT_READY_ERROR:
                        pass  # Brainflow throws this error if no board is found, ignore.
                    else:
                        raise e
                    logging.info("Failed to prepare sessions, trying again.")
            else:  # if board_shim is prepared, keep checking for incoming data.
                data_timestamp = self.board_shim.get_current_board_data(1, BrainFlowPresets.DEFAULT_PRESET)[
                    self.board_shim.get_timestamp_channel(self.board_id, BrainFlowPresets.DEFAULT_PRESET)]
                if len(data_timestamp) > 0:
                    last_timestamp = float(data_timestamp)
                # after timeout of no data received, consider connection dead.
                current_time = time()
                if current_time - last_timestamp > self.time_out:
                    logging.warning("Brainflow session connection time out, trying to reconnect.")
                    self.board_shim.release_session()

    def __del__(self):
        logging.info("Releasing all brainflow sessions.")
        try:
            self.board_shim.release_all_sessions()
        except BrainFlowError as e:
            logging.exception(e)

    def _prepare_board(self) -> None:
        self.board_shim.prepare_session()
        if self.board_id in [BoardIds.MUSE_2_BOARD, BoardIds.MUSE_S_BOARD]:
            self.board_shim.config_board("p50")  # Sets 5th EEG and PPG for Muse 2. Sets 5th EEG for Muse S.
            self.board_shim.config_board("p61")  # Sets PPG for Muse S, only works if p50 is set.
        self.board_shim.start_stream(self.ringbuffer_size, self.streamer_params)
