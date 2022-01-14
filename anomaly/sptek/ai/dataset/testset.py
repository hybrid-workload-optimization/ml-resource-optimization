import time
import pandas as pd

from . import generator as window
from .generator import WindowGenerator
from ..utils import logger, stamp


class TestSet(object):
    
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
        scroll = self.searcher.scroll()
        
        meta = self.master.machine(dtype='forecast')
        delta, unit = stamp.scroll_delta(meta.period, scroll)
        batch = 1
        
        self.log.info(f"WindowGenerator -- delta: {delta}, {unit} -- batch: {batch} -- forecast: test(1)")
                
        return WindowGenerator(int(delta),
                               int(delta),
                               int(delta),
                               None,
                               None,
                               self.dataset,
                               label_columns=self.dataset.columns,
                               batch_size = batch)
    
    def __repr__(self):
        return repr(self.dataset)
    
    