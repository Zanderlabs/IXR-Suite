import logging
import time
from asyncio import Event
from threading import Thread

from brainflow import BoardShim
from pylsl import (StreamInfo, StreamInlet, StreamOutlet, local_clock,
                   resolve_byprop)
from z_flow.classifiers import Classifier, ClfError


class DecodeError(Exception):
    pass


class LslEventListener(Thread):
    """Class that listens to incoming LSL events. Per incoming LSL event it starts
    a child worker (thread) that will decode the message and depending on the content
    will create a classifier instance, collect data, train and/or predict the models.
    Instances of this class are executed in it's own thread of control.

    The instance will automatically shutdown if the stay_alive event has been cleared.

    Extends from threading.Thread, for more information:
    https://docs.python.org/3/library/threading.html#thread-objects

    If thread_daemon the parameter is set, the thread is launched in daemon mode,
    the significance of this flag is that a deamon thread does not keep the Python process alive.

    :param board_shim: Brainflow BoardShim to collect data from EEG devices.
    :type board_shim: BoardShim
    :param stay_alive: Life line to indicate that the thread should stay alive.
    :type stay_alive: Event
    :param thread_name: Thread name, defaults to "lsl_event_listener"
    :type thread_name: str, optional
    :param thread_daemon: Sets thread as daemon, or not, defaults to True
    :type thread_daemon: bool, optional
    """

    def __init__(self, board_shim: BoardShim, stay_alive: Event,
                 thread_name: str = "lsl_event_listener", thread_daemon: bool = False) -> None:
        Thread.__init__(self, name=thread_name, daemon=thread_daemon)
        self.stay_alive = stay_alive
        self.board_shim = board_shim
        self.classifiers = {}
        name = 'z-flow-lsl-relay'
        logging.info(f"Starting '{name}' LSL event relay stream.")
        self.outlet = StreamOutlet(StreamInfo(name, 'Markers', 1, 0, 'string', 'z-flow-lsl-relay'))
        logging.info(f"'{self.outlet.get_info().name()}' LSL event relay stream started.")

    def run(self) -> None:
        """Once a thread object is created, its activity must be started by calling the thread’s start() method.
        This invokes the run() method in a separate thread of control.
        """
        threads = []
        connections = []
        inlet = None

        while self.stay_alive.is_set() and len(connections) == 0:
            # Poll for connections as long the thread is alive.
            # Make sure resolve_byprop has a timeout, otherwise the thread hangs.
            connections = resolve_byprop("name", "SendMarkersOnClick", timeout=1.0)

        if len(connections) > 0:  # Only build connection when previous loop was exited because a connection was found.
            logging.info(f"LSL event stream found: {connections[0].name()}, connecting ...")
            inlet = StreamInlet(connections[0])
            logging.info(f"LSL event stream established: {inlet.info().name()}")

        while self.stay_alive.is_set() and inlet is not None:
            # Poll for incoming events as long the thread is alive.
            # Make sure pull_sample has a timeout, otherwise the thread hangs.
            event_sample, event_timestamp = inlet.pull_sample(timeout=1.0)
            if event_sample is not None:
                thread = Thread(target=self._event_worker, args=(event_sample, event_timestamp))
                threads.append(thread)
                thread.start()

        # Once stay_alive is cleared, wait for threads to finish
        for thread in threads:
            thread.join()

    def _event_worker(self, event_sample: list[str], event_timestamp: float) -> None:
        """Thread worker that will handle/decode and incoming LSL and
        takes the appropriate actions based on the event.

        :param event_sample: List with possible event samples, will always take the first sample to decode.
        :type event_sample: List of str
        :param event_timestamp: The original event timestamp
        :type event_timestamp: float
        """
        lsl_local_time = local_clock()
        local_time = time.time()
        event_timestamp = event_timestamp + (local_time - lsl_local_time)

        logging.info(f"LSL event received, timestamps: "
                     f"\n\tLSL local timestamp: \t{lsl_local_time}"
                     f"\n\tLocal timestamp: \t{local_time}"
                     f"\n\tEvent timestamp: \t{event_timestamp}")

        try:
            self._message_decode(event_sample[0], event_timestamp)
        except (DecodeError, ClfError) as e:
            logging.warning(f"{e}. Stopping thread, please try again.")

    def _message_decode(self, message: str, event_timestamp: float) -> str:
        """Parsers, decodes and executes LSL events passed as message.
        Returns a success notification or raises a DecodeError.

        A description of lsl can be found at: `./doc/lsl.md`

        :param message: LSL event encoded as str.
        :type message: str
        :param event_timestamp: The original event timestamp
        :type event_timestamp: float
        :raises DecodeError: "Unknown classifier instance, please create one"
        :raises DecodeError: "Unrecognized task when decoding"
        :return: Returns success message
        :rtype: str
        """
        if message == '':
            raise DecodeError("Got empty event message when decoding.")
        message_list = message.split(';')
        task = message_list.pop(0)
        name = message_list.pop(0)
        if task == 'create':
            model_type = message_list.pop(0)
            time_range = [int(value) for value in message_list.pop(0).split(',')]
            filter_freq_cutoff = [float(value) for value in message_list.pop(0).split(',')]
            method = message_list.pop(0)
            self.classifiers[name] = Classifier(self.board_shim, model_type, time_range, filter_freq_cutoff, method)
            logging.info(f"Created classifier instance, with name {name}.")
        elif task == 'collect' and name in self.classifiers:
            label = int(message_list.pop(0))
            self.classifiers[name].collect_sample(label, event_timestamp)
            logging.info(f"Collected sample with, label: {label}.")
        elif task == 'train' and name in self.classifiers:
            scores = self.classifiers[name].train()
            logging.info(f"Trained model successfully, with scores:.")
            for key, value in scores.items():
                logging.info(f"    {key}: {value}")
        elif task == 'predict' and name in self.classifiers:
            prediction, probabilities = self.classifiers[name].predict(event_timestamp)
            logging.info(f"Prediction: {prediction}, with probabilities: {probabilities}")
        elif name not in self.classifiers:
            raise DecodeError("Unknown classifier instance, please create one.")
        else:
            raise DecodeError("Unrecognized task when decoding.")
