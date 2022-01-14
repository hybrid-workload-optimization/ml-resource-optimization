import sys
import time
import argparse
import tensorflow as tf
from pathlib import Path

import anomaly
from ai.utils import logger, ThreadPool

# create command parser
parser = argparse.ArgumentParser(prog="model train service", description="the controller that controls the company's ai service.")
subparser = parser.add_subparsers(dest="command")

# add auto command parser
commander = subparser.add_parser('anomaly', help='Run the predictive model and return predicted values.')
commander.add_argument('master', type=str, help='Input forcat config yaml file.')
commander.add_argument('-w', '--worker', type=int, default=3, help='Enter the number of workers.')
commander.add_argument('-l', '--log', type=str, default='estimator-ml.log', help='Specify the location where the log will be saved.')
commander.add_argument('-v', '--verbose', type=int, default='1', help='Specifies the level of log generation.')





def main(*args):
    executer = parser.parse_args(*args)

    if executer.command == "anomaly":
        anomaly.execute(executer)

if __name__ == "__main__":
    main(sys.argv[1:])