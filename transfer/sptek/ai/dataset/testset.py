import time
import pandas as pd

from . import generator as window
from .generator import WindowGenerator
from ..utils import logger, stamp


class TestSet(object):
    
    def __init__(self, master, dataset):
        self.log = logger.get(self)
        self.master = master
        self.dataset = dataset

    def fit(self):
        pass
        
    def generator(self):
        meta = self.master.machine(dtype='forecast')
        delta, unit = stamp.get_delta(meta.term)
        batch = 1
        
        meta = self.master.machine(dtype='features')
        columns = meta.output
        
        self.log.info(f"WindowGenerator -- delta: {delta}, {unit} -- batch: {batch} -- forecast: test(1)")
        
        # return WindowGenerator(int(delta),
        #                        int(delta),
        #                        int(delta),
        #                        None,
        #                        None,
        #                        self.dataset[columns],
        #                        label_columns=columns,
        #                        batch_size = batch)
        
        return self.dataset[columns].values.reshape(1, delta, len(columns))
    
    def columns(self):
        meta = self.master.machine(dtype='features')
        return meta.output
    
    def __repr__(self):
        return repr(self.dataset)
    
    