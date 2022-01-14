import time
import pandas as pd

from . import generator as window
from .generator import WindowGenerator
from ..utils import logger
from ..utils import stamp


class TrainSet(object):
    
    def __init__(self, master, dataset):
        self.log = logger.get(self)
        self.master = master
        self.dataset = dataset

    def fit(self):
        pass
        
    def generator(self):
        meta = self.master.machine(dtype='forecast')
        delta, unit = stamp.get_delta(meta.term)

        meta = self.master.machine(dtype='hyperparameter')
        batch = meta.batch
        
        meta = self.master.machine(dtype='sampling')
        train_size = meta.train
        val_size = meta.val
        
        meta = self.master.machine(dtype='features')
        columns = meta.input
        
        self.log.info(f"WindowGenerator -- delta: {delta}, {unit} -- batch: {batch} -- sampling: train({train_size}), val({val_size})")
        
        train_df, val_df, test_df = window.split_data(self.dataset[columns], train_size, val_size)
        return WindowGenerator(int(delta),
                               int(delta),
                               int(delta),
                               train_df,
                               val_df,
                               test_df,
                               label_columns=train_df.columns,
                               batch_size = batch)
        
    def __repr__(self):
        return repr(self.dataset)
    