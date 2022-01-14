import pandas as pd
from operator import itemgetter
from ..utils import logger, stamp, TimeGenerator

import threading
locker = threading.Lock()

# import re
# tick_map = {
#     'us': 'microseconds',
#     'ms': 'milliseconds',
#     's': 'seconds',
#     'm': 'minutes',
#     'h': 'hours',
#     'd': "days",
#     'w': "weeks"
# }

# def _start_time_(t, f):
#     tick = re.sub(r'[^a-z]', '', f)
#     numbers = re.sub(r'[^0-9]', '', f)
#     return TimeGenerator.prev(t, tick_map[tick], int(numbers))
    

# def _next_time_(t, f):
#     tick = re.sub(r'[^a-z]', '', f)
#     numbers = re.sub(r'[^0-9]', '', f)
#     return TimeGenerator.next(t, tick_map[tick], int(numbers))
    
    
class TermSearcher(object):

    def __init__(self, master, embed, scroll):
        self.log = logger.get(self)
        self.master = master
        self.embed = embed
        self.scroll = scroll
        self._iter_current_ = 0
        self._iter_start_ = 0
        self._iter_end_ = 0
        self.fit()

    def fit(self):
        _series_ = self.series()
        self.first = _series_.min()
        self.last = _series_.max()
        self._iter_current_ = self.first
        self.log.info(f"{self.first} ~ {self.last}")
        self._iter_start_ = self._iter_end_ = self._iter_current_
        
    def series(self):
        self._series_ = pd.Series(self.embed['log_time'], name='log_time')
        self._series_ = pd.to_datetime(self._series_)
        return self._series_

    def count(self):
        return self.embed.count()
    
    def current(self):
        return self.embed.current()
    
    def ago(self):
        return self.embed.ago()
    
    def interval(self):
        return self.embed.interval()
            
    def __iter__(self):        
        return self

    def __next__(self):

        with locker:
            if self._iter_end_ >= self.current():
                raise StopIteration
            
            self._iter_end_ = stamp.next_time(self._iter_end_, self.scroll)
            _i_s_ = self._iter_start_
            _i_e_ = self._iter_end_
            self._iter_start_ = self._iter_end_
            
        a = _i_s_
        b = self.current()
        if a >= b :
            raise StopIteration

        try:
            timeset = self.series()
            timeslice = (timeset >= _i_s_) & (timeset <= _i_e_)
            _iter_ = timeslice.index[timeslice].tolist()
            
            if len(_iter_) == 0:
                return (_i_s_, _i_e_, 0, [], [], [], [])
        
            t = itemgetter(*_iter_)(self.embed['log_time'])
            m = itemgetter(*_iter_)(self.embed['msg_id'])
            d = itemgetter(*_iter_)(self.embed['dense_vector'])
            return (
                min(t),
                max(t),
                len(_iter_),
                _iter_,
                t,
                m,
                d
            )
            
        except:
            raise StopIteration

class BucketSearcher(object):
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.current = dataset.current
        self.ago = dataset.ago
        self.interval = dataset.interval
        self.bucket_series = dataset.bucket_series()
        self.index = -1
    
    def count(self):
        return self.bucket_series.count()
    
    def cluster(self):
        return self.bucket_series.n_cluster
    
    def scroll(self):
        return self.bucket_series.scroll
    
    def __getitem__(self, index):
        return self.bucket_series[index]
    
    def __iter__(self):        
        return self

    def __next__(self):
        try:
            self.index = self.index + 1
            return self.bucket_series[self.index]
        except:
            raise StopIteration
        
    