import datetime
import pandas as pd

from ..utils import logger

class Generator(object):
    
    def transform(self, dataset):
        raise NotImplementedError

class WeekdayGenerator(Generator):
    
    def __init__(self, master):
        self.log = logger.get(self)
        self.master = master
        
        meta = master.machine(dtype='features')
        self.key = meta.key
        
        meta = master.machine(dtype='generators')
        self.col = meta.feature
        
    def transform(self, dataset):
        if not self.key in dataset.columns:
            raise Exception(f"Not found column '{self.key}' on pandas.DataFrame.")
        
        weekday = list(map(datetime.date.isoweekday, dataset[self.key]))
        dataset.append(self.col, weekday, axis=1)
        return dataset
