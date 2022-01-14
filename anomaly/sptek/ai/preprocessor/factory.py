
from ..utils import logger
from .translate import Translate
from .scaler import StandardScaler


class PreprocessorFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master, dtype='token'):
        meta = master.preprocess(dtype=dtype)
        # logger.debug(meta)

        if meta.type == "token":
            return Translate(meta)
        

class ScalerFactory(object):

    @classmethod
    def create(cls, master, dtype='scaling'):
        meta = master.machine(dtype=dtype)
        # logger.debug(meta)

        if meta.algorithm == "z-score":
            return StandardScaler(master, meta)