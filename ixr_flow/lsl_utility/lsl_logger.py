from logging import StreamHandler

from pylsl import StreamInfo, StreamOutlet


class LslLogger(StreamHandler):
    """StreamHandler child class to log messages over LSL.

    :param name: Identifying name to be used for this LSL stream, defaults to 'lsl_logger'
    :type name: str, optional
    :param stype: Type of messages to be pushed over the LSL stream, defaults to 'log'
    :type stype: str, optional
    """

    def __init__(self, name: str = 'lsl_logger', stype: str = 'log') -> None:
        StreamHandler.__init__(self)
        self.outlet = StreamOutlet(StreamInfo(name=name, type=stype, channel_count=1,
                                   channel_format='string', source_id='ixr-flow-lsl-logger'))

    def emit(self, message: str) -> None:
        """Method that is called by the logger to log a messages.

        :param message: _description_
        :type message: str
        """
        msg = self.format(message)
        self.outlet.push_sample([msg])
