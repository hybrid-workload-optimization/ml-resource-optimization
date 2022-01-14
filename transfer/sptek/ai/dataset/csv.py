import pandas as pd
from re import S

from .data import TFData
from ..utils import logger


class CSVData(TFData):
    
    def __init__(self, master, meta, key='DATE'):
        super().__init__(master, meta, key)
        self.log = logger.get(self)
        
    def load(self, path):
        self._data_ = pd.read_csv(path, sep=',')
        
        if not self.key in self._data_.columns:
            raise Exception(f"Not found column '{self.key}' on pandas.DataFrame.")
        
        self.to_number(self.meta.columns)
        self.to_datetime(self.key)
                
        # logger.debug(f"\n{self._data_.info()}")
        