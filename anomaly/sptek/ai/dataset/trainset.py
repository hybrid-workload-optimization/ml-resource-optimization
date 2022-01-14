import time
import pandas as pd

from . import generator as window
from .generator import WindowGenerator
from ..utils import logger, stamp


class TrainSet(object):
    
    def __init__(self, master, searcher):
        self.log = logger.get(self)
        self.master = master
        self.searcher = searcher

    def fit(self):
        start = time.time()
        
        self.category =[]
        n_cluster = self.searcher.cluster()
        
        for i in range(n_cluster):
            self.category.append(pd.Series(name=f"categor.{i}"))
            
        for bucket in self.searcher:
            for item in bucket.categories:
                a = self.category[item.label]
                b = pd.Series([item.count()], name=f"categor.{item.label}")                
                self.category[item.label] = a.append(b)
        
        self.dataset = pd.concat(self.category, axis=1, ignore_index=False)
        
        duration =  time.time() - start        
        self.log.info(f"build train dataset completed.-- {self.dataset.shape} -- (duration={duration:.5f}s)")
        
    def generator(self):
        meta = self.master.machine(dtype='forecast')
        delta, unit = stamp.scroll_delta(meta.period, meta.term)

        meta = self.master.machine(dtype='hyperparameter')
        batch = meta.batch
        
        meta = self.master.machine(dtype='sampling')
        train_size = meta.train
        val_size = meta.val
        
        self.log.info(f"WindowGenerator -- delta: {delta}, {unit} -- batch: {batch} -- sampling: train({train_size}), val({val_size})")
        
        train_df, val_df, test_df = window.split_data(self.dataset, train_size, val_size)
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
    
    