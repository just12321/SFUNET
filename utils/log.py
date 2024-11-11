from __future__ import annotations
import logging
import io
from datetime import datetime
import time

from tqdm import tqdm
from .Accumulator import Accumulator
from torch.utils.tensorboard import SummaryWriter
import numpy as np

LOGGER_NAME = 'root'
LOGGER_DATEFMT = '%Y-%m-%d %H:%M:%S'

handler = logging.StreamHandler()

logger = logging.getLogger(LOGGER_NAME)
# logger.addHandler(handler)
logging.basicConfig(level=logging.INFO, format='(%(levelname)s) %(asctime)s: %(message)s', datefmt=LOGGER_DATEFMT)

def add_logging(logs_path, prefix):
    log_name = prefix + datetime.strftime(datetime.today(), '%Y-%m-%d_%H-%M-%S') + '.log'
    stdout_log_path = logs_path / log_name

    fh = logging.FileHandler(str(stdout_log_path))
    logger.addHandler(fh)

class TqdmHandler(logging.Handler):
    def __init__(self, level=logging.INFO):
        super(TqdmHandler, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger:logging.Logger, level = logging.INFO, mininterval = 5):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level
        self.mininterval = mininterval
        self.last_time = 0

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        if len(self.buf) > 0 and (t:=time.time()) - self.last_time > self.mininterval:
            self.logger.log(self.level, self.buf)
            self.last_time = t
    
class SummaryWriterAvg(SummaryWriter):
    def __init__(self, *args, dump_period = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.dump_period = dump_period
        self._avg_scalars = dict()

    def add_scalar(self, tag, value, global_step=None, disable_avg = False):
        if disable_avg or isinstance(value,(tuple, list, dict)):
            super().add_scalar(tag, np.array(value), global_step)
        else:
            if tag not in self._avg_scalars:
                self._avg_scalars[tag] = Accumulator()
            avg_scalar:Accumulator = self._avg_scalars[tag]
            avg_scalar += value
            if avg_scalar.count >= self.dump_period:
                super().add_scalar(tag, avg_scalar.avg, global_step)
                avg_scalar.reset()