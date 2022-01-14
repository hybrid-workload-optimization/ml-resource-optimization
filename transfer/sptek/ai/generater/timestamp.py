import pandas as pd
from ..utils import logger
from ..utils import stamp

class Timestamp(object):
    
    def __init__(self, master, time, meta):
        self.log = logger.get(self)
        self.master = master
        self.time = time
        self.meta = meta
    
    def generator(self):
        t, _ = stamp._tick_and_nubmer_(self.meta.term)
        freq = '1' + t
        start_time = stamp.next_time(self.time, freq)
        end_time = stamp.next_time(start_time, self.meta.term)
        return pd.date_range(start=start_time, end=end_time, freq=freq).values
    
    def columns(self):
        return self.meta.key