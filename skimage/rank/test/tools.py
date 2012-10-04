__author__ = 'Olivier Debeir 2021'

import logging
import time

def init_logger(logfilename = 'myapp.log'):
    """add logger capabilities
    """
    FORMAT = '%(asctime)-15s %(processName)s %(process)d %(message)s'
    logging.basicConfig(filename=logfilename,format=FORMAT,filemode='wt')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # add ch to logger
    logger.addHandler(ch)
    logger.info('start logging in %s' % logfilename)
    return logger


logger = logging.getLogger()

def log_timing(func):

    def wrapper(*arg):
        log_timing.level += 1
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        ms = (t2-t1)*1000.0
        logger.info('%s%s took %0.3f ms' % (log_timing.level*'-',func.func_name, ms))
        log_timing.level -= 1
        return (res,ms)

    return wrapper

log_timing.level = 0


