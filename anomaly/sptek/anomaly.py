import json
import string
from pathlib import Path

# import tensorflow as tf
# import ai.gpus as gpus
from ai.master import Master
from ai.utils import logger, ThreadPool #, serialize
# from ai.searcher import SearcherFactory
# from ai.model import ModelFactory
# from ai.dataset import CMPDataSet
from ai.rest import RESTful
from ai import service
from ai.scheduler import SchedulerFactory

def execute(executer):
    logger.debug(f"executer: {executer}")
    logger.debug(f"executer: {executer.master}")

    if executer.command != "anomaly":
        return

    logger.verbose(executer.verbose)
    logger.file(executer.log)
    logger.createFileHandler()
    
    # load master file
    master = Master(executer.master)
    master.load()
    master.work_dir = Path(__file__).parent.parent.absolute()

    # initailize thread pool
    pool = ThreadPool()
    pool.createPool(executer.worker)
    pool.start()
    
    # setup service
    # It is Router connection option. If True , it is used, if False it is not used.
    master._use_route_ = executer.route  
    service.v1.setup(master)

    # # initailize scheduler
    # job = service.v1.create(master)
    # scheduler = SchedulerFactory.create(master)
    # scheduler.add_job(job)
    # scheduler.start()

    # RESTful API 서비스 실행
    rest = RESTful()
    rest.api.add_namespace(service.v1.Train, '/v1')
    rest.api.add_namespace(service.v1.Forecast, '/v1')
    rest.start(master.port())
    rest.join()
