
import logging
import coloredlogs
from datetime import datetime


# Global variables for logging
LOG_TO_CONSOLE = True
CONSOLE_LEVEL = 'INFO'
LOG_TO_FILE = True
FILE_LEVEL = 'INFO'

class LoggerMixin():
    @property
    def logger(self):
        return logging.getLogger(
            "{}.{}".format(type(self).__module__, type(self).__name__)
        )

def setup_logger():
    logger = logging.getLogger()
    # level_styles = {'warning': {'color': 'red', 'bold': False}}
    fmt_console = '%(name)s %(levelname)s %(message)s'
    fmt_file = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    file_name = 'floris_{:%Y-%m-%d-%H_%M_%S}.log'.format(datetime.now())

    if LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(CONSOLE_LEVEL)
        console_format = coloredlogs.ColoredFormatter(
            # level_styles=level_styles,
            fmt=fmt_console)
        console_handler.setFormatter(console_format)
        console_handler.addFilter(TracebackInfoFilter(clear=True))
        logger.addHandler(console_handler)

    if LOG_TO_FILE:
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(FILE_LEVEL)
        file_format = logging.Formatter(fmt_file)
        file_handler.setFormatter(file_format)
        file_handler.addFilter(TracebackInfoFilter(clear=False))
        logger.addHandler(file_handler)

    return logger

class TracebackInfoFilter(logging.Filter):
    """Clear or restore the exception on log records"""

    def __init__(self, clear=True):
        self.clear = clear

    def filter(self, record):
        if self.clear:
            record._stack_info_hidden, record.stack_info = \
                                                        record.stack_info, None
        elif hasattr(record, "_stack_info_hidden"):
            record.stack_info = record._stack_info_hidden
            del record._stack_info_hidden
        return True
