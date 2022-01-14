import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ai import gpus
gpus.set_device_configuration(size=256)

import sys
import argparse
import transfer

# create command parser
parser = argparse.ArgumentParser(prog="model train service", description="the controller that controls the company's ai service.")
subparser = parser.add_subparsers(dest="command")

# add auto command parser
commander = subparser.add_parser('transfer', help='Run the predictive model and return predicted values.')
commander.add_argument('master', type=str, help='Input forcat config yaml file.')
commander.add_argument('-w', '--worker', type=int, default=3, help='Enter the number of workers.')
commander.add_argument('-l', '--log', type=str, default='estimator-ml.log', help='Specify the location where the log will be saved.')
commander.add_argument('-v', '--verbose', type=int, default='1', help='Specifies the level of log generation.')
commander.add_argument('--route', dest='route', action='store_true', help='Use router connection.')
commander.add_argument('--no-route', dest='route', action='store_false', help='Not use router connection.')
commander.set_defaults(route=True)

    
def main(*args):
    executer = parser.parse_args(*args)
    
    if executer.command == "transfer":
        transfer.execute(executer)


if __name__ == "__main__":
    main(sys.argv[1:])
