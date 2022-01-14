from pathlib import Path

from ai import service
from ai.master import Master
from ai.rest import RESTful
from ai.utils import logger
from ai.utils import ThreadPool

def execute(executer):
    logger.debug(f"executer: {executer}")
    logger.debug(f"executer: {executer.master}")

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

    # RESTful API 서비스 실행
    rest = RESTful()
    rest.api.add_namespace(service.v1.Forecast, '/v1')
    rest.start(master.port())
    rest.join()
