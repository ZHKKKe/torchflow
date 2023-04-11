import logging


format_str = '%(message)s'
formatter = logging.Formatter(format_str)
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger('TorchFlow')


class MODE:
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    WARN = logging.WARN
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET


def mode(mode=MODE.INFO):
    global logger
    logger.setLevel(mode)


def tofile(fpath, mode=MODE.INFO):
    global logger
    global formatter

    fh = logging.FileHandler(fpath)
    fh.setLevel(mode)
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)


def log(message):
    global logger
    
    out = message
    if isinstance(message, list):
        out = ''.join(message)
    
    logger.info(out)


def info(message):
    global logger
    
    out = message
    if isinstance(message, list):
        out = ''.join(message)
    out = '\n' + '=' * 40 + ' INFO ' + '=' * 40 + '\n' + out + '=' * 86
    
    logger.info(out)


def warn(message):
    global logger

    out = message
    if isinstance(message, list):
        out = ''.join(message)
    out = '\n' + '=' * 40 + ' WARN ' + '=' * 40 + '\n' + out + '=' * 86
                                                    
    logger.warn(out)


def error(message):
    global logger

    out = message
    if isinstance(message, list):
        out = ''.join(message)
    out = '\n' + '=' * 40 + ' ERROR ' + '=' * 39 + '\n' + out + '=' * 86

    logger.error(out)
    exit()
