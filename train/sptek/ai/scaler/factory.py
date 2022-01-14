
from .standard import StandardScaler
from ..utils import logger

class ScalerFactory(object):
    
    @classmethod
    def create(cls, master):
        meta = master.machine(dtype='scaler')
        # logger.debug(f"{meta}")
        
        if meta.algorithm == 'z-score' or meta.algorithm == 'StandardScaler':
            return StandardScaler(master)

        log = logger.get(cls)
        log.info(f"skip scaler transform.")
        return None
        
