
from .isolationiorest import IsolationForest
from ..utils import logger

class OutlierFactory(object):
    
    @classmethod
    def create(cls, master):
        meta = master.machine(dtype='outlier')
        # logger.debug(f"{meta}")
        
        if meta.algorithm == 'IsolationForest':
            return IsolationForest(master)

        log = logger.get(cls)
        log.info(f"skip outiler dection.")
        return None
