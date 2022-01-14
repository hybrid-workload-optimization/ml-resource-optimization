import pandas as pd

from .data import TFData
from ..utils import logger
from ..utils import stamp


class DataSearcher(object):
    
    def __init__(self, master, dataset, res, action):
        self.log = logger.get(self)
        self.master = master
        self.dataset = dataset
        self.action = action
        self.res = res
        
    def filter(self, target, request_time):        
        if self.action == 'train':
            meta = self.master.machine(dtype='sequential')
        else:
            meta = self.master.machine(dtype='forecast')
        
        # end_time = stamp.next_time(request_time, meta.term)
        end_time = stamp.strptime(request_time)
        start_time = stamp.prev_time(request_time, meta.term)
        
        self.log.info(f"model: {target} -- filter: {start_time} ~ {end_time}")
        return self._lookup_(target, start_time, end_time)
        
    def _lookup_(self, target, start_datetime, end_datetime):        
        mask = (self.dataset.date > start_datetime) & (self.dataset.date <= end_datetime)
        lookup = self.dataset[mask]
    
        mask = lookup[self.res.column] == target
        lookup = lookup[mask]
        
        obj = TFData(self.master, self.dataset.meta, self.dataset.key)
        obj._data_ = lookup
        return obj
        