import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ai import gpus
gpus.set_device_configuration(size=1024 * 3)

import sys
import time
import argparse
from pathlib import Path
    
import anomaly
from ai.utils import logger, ThreadPool, passwd


# create command parser
parser = argparse.ArgumentParser(prog="model train service", description="the controller that controls the company's ai service.")
subparser = parser.add_subparsers(dest="command")

# add auto command parser
commander = subparser.add_parser('anomaly', help='Run the predictive model and return predicted values.')
commander.add_argument('master', type=str, help='Input forcat config yaml file.')
commander.add_argument('-w', '--worker', type=int, default=3, help='Enter the number of workers.')
commander.add_argument('-l', '--log', type=str, default='estimator-ml.log', help='Specify the location where the log will be saved.')
commander.add_argument('-v', '--verbose', type=int, default='1', help='Specifies the level of log generation.')
commander.add_argument('--route', dest='route', action='store_true', help='Use router connection.')
commander.add_argument('--no-route', dest='route', action='store_false', help='Not use router connection.')
commander.set_defaults(route=True)

# add auto command parser
commander = subparser.add_parser('passwd', help='Run the predictive model and return predicted values.')
commander.add_argument('crypto', type=str, help='Input forcat config yaml file.')
commander.add_argument('pwd', type=str, help='Input forcat config yaml file.')
commander.add_argument('-key', type=str, default='', help='Input forcat config yaml file.')
commander.add_argument('--random-key', dest='randomkey', action='store_true', help='Not use router connection.')
commander.add_argument('--no-random-key', dest='randomkey', action='store_true', help='Not use router connection.')
commander.add_argument('-hash', type=str, default='sha256', help='Input forcat config yaml file.')
commander.set_defaults(randomkey=True)



def main(*args):
    executer = parser.parse_args(*args)
    
    if executer.command == "anomaly":
        anomaly.execute(executer)
        
    if executer.command == "passwd":
        if executer.crypto == 'encode':
            passwd.encrypto(executer.pwd, executer.key, executer.hash, executer.randomkey)
        
        elif executer.crypto == 'decode':
            passwd.decrypto(executer.pwd, executer.key)
            

if __name__ == "__main__":
    main(sys.argv[1:])
