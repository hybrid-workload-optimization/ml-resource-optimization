import inspect
import logging
import logging.handlers
import os
import six
import sys

from . import stamp

formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')

levels = [
    logging.DEBUG ,
    logging.INFO ,
    logging.WARNING ,
    logging.ERROR ,
    logging.CRITICAL
]

instances = {}

logVerbose = 1
logFile = None
logFileHandler = None

def file(path):
    global logFile
    logFile = os.path.abspath(path)

def verbose(verbose):
    global logVerbose
    logVerbose = max(min(verbose, 4), 0)

def path():
    if logFile is None:
        return f"{os.path.expanduser('~')}/{stamp.nowstamp()}.log"
    return logFile

def createFileHandler():
    global logFileHandler
    logFileHandler = fileHandler(path(), level(logVerbose))

def level(verbose):
    return levels[verbose]

def streamHandler(level):
    try:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(level)
        return handler
    except:
        return None


def fileHandler(path, level):
    try:
        # handler = logging.FileHandler(path) #, mode='w'
        handler = logging.FileHandler(path, mode='w')
        handler.setFormatter(formatter)
        handler.setLevel(level)
        return handler
    except:
        return None

def lookup(name):
    for key in list(instances.keys()):
        if key == name:
            return instances[key]
    return None

def get(name):
    if not isinstance(name, str):
         name = name.__class__.__name__
    
    log = lookup(name)
    if log is not None:
        return log

    log = logging.getLogger(name)    
    log.addHandler(streamHandler(level(logVerbose)))
    # log.addHandler(fileHandler(path(), level(logVerbose)))
    
    if logFileHandler is not None:
        log.addHandler(logFileHandler)
    else:
        log.addHandler(fileHandler(path(), level(logVerbose)))
        
    log.setLevel(level(logVerbose))
    logging.root = log
    instances[name] = log
    return log


def debug(msg):
    print(f"(debug) {msg}")