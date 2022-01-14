from .power import PowerScaler
from .standard import StandardScaler
from ..prefix import ScalerPath
from ..utils import logger

class ScalerFactory(object):
    
    @classmethod
    def load(cls, master):
        log = logger.get(cls)
        
        meta = master.import_model(dtype='scaler', lookup="match")
        if not meta:
            return None
        
        scaler = None
        if meta.algorithm == 'z-score' or meta.algorithm == 'StandardScaler':
            scaler = StandardScaler(master)
            
        if meta.algorithm == 'log' or meta.algorithm == 'PowerScaler':
            scaler = PowerScaler(master)
        
        if not scaler:
            log.info(f"not support scaler transform algorithm.")
                
        target = ScalerPath(master)
        scaler.load(target.path())
        return scaler
        
    @classmethod
    def create(cls, master):
        log = logger.get(cls)
        
        scaler = cls.load(master)
        if scaler:
            return scaler
    
        meta = master.machine(dtype='scaler')
        # logger.debug(f"{meta}")
        
        if meta.algorithm == 'z-score' or meta.algorithm == 'StandardScaler':
            scaler = StandardScaler(master)
            return scaler
        
        if meta.algorithm == 'log' or meta.algorithm == 'PowerScaler':
            scaler = PowerScaler(master)
            return scaler

        log.info(f"skip scaler transform.")
        return None
        
