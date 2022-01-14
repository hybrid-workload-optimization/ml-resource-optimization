from re import L
import time
import string
import pandas as pd

from ..utils import logger, ThreadPool, thread, stamp
from ..scheduler import JobTrigger

import threading
locker = threading.Lock()


class CMPDataSet(JobTrigger):

    def __init__(self, searcher=None, preprocessor=None):
        self.log = logger.get(self)
        self._searcher_ = searcher
        self._preprocessor_ = preprocessor
        self._dataset_ = []
        self._iter_index_ = 0
        self.session = []
        self._session_completed_ = False

        self._time_ = []
        self._msg_id_ = []
        self._embed_ = []
        self._shape_ = 0
        
    def build(self, id = 0):
        for _size_, _start_, _end_, _time_, _msg_id_, _dense_ in self.searcher():
            self._time_.extend(_time_[_start_:_end_])
            self._msg_id_.extend(_msg_id_[_start_:_end_])
            self._embed_.extend(_dense_)
            self._shape_ = self._shape_ + _size_
        
    def data(self):
        return self._dataset_

    def filter(self, start, end):
        self._start_time_ = start
        self._end_time_ = end
        self.searcher().search(start, end)

    def searcher(self):
        return self._searcher_

    def preprocessor(self):
        return self._preprocessor_

    def messages(self):
        return self.data()['message']

    def append(self, ds):
        df = pd.DataFrame.from_dict(ds)
        self._dataset_ = pd.concat([self._dataset_, df], axis=1)
        
    def copy(self):
        clone = CMPDataSet(self.searcher(), self.preprocessor())
        clone._dataset_ = self._dataset_.copy()
        clone._start_time_ = self._start_time_
        clone._end_time_ = self._end_time_
        return clone

    def shape(self):
        return (self._shape_, 3)

    def sort(self):
        self._dataset_ = self._dataset_.sort_index()

    def drop_duplicates(self):
        self._dataset_ = self._dataset_.groupby(level=0).last()

    def set_index(self, name='time'):
        self._dataset_ = self._dataset_.set_index(name, drop=True)

    def fork_build(self, fork=1):
        pool = ThreadPool()
        for _ in range(fork):
            self.session.append(pool.add_task(self._fork_serve_))
        pool.add_task(self._check_fork_)

    def _fork_serve_(self, session=None):
        self.build(session)

    def _check_fork_(self, session=None):
        start = time.time()
        self._wait_completed_()
        duration =  time.time() - start
        self.log.info(f"all subprocess for build the data set completed. -- {self.shape()} -- (duration={duration:.5f}s)")
        self._session_completed_ = True

    def _wait_completed_(self):
        with locker:
            pool = ThreadPool()
            _del_ = []
            for s in self.session:
                pool.wait_completion(s)
                _del_.append(s)
            for d in _del_:
                self.session.remove(d)
        
    def wait_completed(self):
        while not self._session_completed_:
            thread.nsleep(1)

    def count(self):
        return self._shape_
    
    def current(self):
        return self._searcher_.current()
    
    def ago(self):
        return self._searcher_.ago()
    
    def interval(self):
        return self._searcher_.interval()
    
    def job(self):
        start = time.time()
        
        inputs = self.count()
        self.log.debug(f"job -- dataset -- (shape={self.shape()})")
        
        pool = ThreadPool()
        self.session.append(pool.add_task(self._fork_serve_))
        self._wait_completed_()
        
        inputs = self.count() - inputs
    
        # check out old data.
        _ago_ = stamp.prev_time(self.searcher().current(), self.searcher().ago())
        _times_ = pd.to_datetime(pd.Series(self._time_))
        _filter_ = _times_ <= _ago_
        indexs = _times_.index[_filter_].tolist()
        indexs.sort(reverse=True)
        
        _del_t_str_ = 'no match'
        remove_size = len(indexs)
        if remove_size:
            self._shape_ = self._shape_ - remove_size
            _del_times_ = [ self._time_[x] for x in indexs ]
            _del_times_ = pd.to_datetime(pd.Series(_del_times_))
            _del_t_str_ = f"{_del_times_.min()} ~ {_del_times_.max()}"
            del _del_times_
        
        # remove element
        for d in indexs:
            del self._time_[d]
            del self._msg_id_[d]
            del self._embed_[d]
            
        duration =  time.time() - start
        self.log.debug(f"job -- dataset -- until time = {_times_.min()} ~ {_times_.max()}")
        self.log.info(f"job -- dataset -- input={inputs}, remove={remove_size}, shape={self.shape()} -- search= ~{self.current()} -- remove= ~{_ago_} -- (duration={duration:.5f}s)")



    def __iter__(self):
        return self

    def __next__(self):

        while self.count() <= self._iter_index_ and not self._session_completed_:
            thread.usleep(1)
            # thread.msleep(1)
            
        self.log.debug(f"count={self.count()}")

        _iter_start_ = self._iter_index_
        _iter_end_ = self.count()
        self._iter_index_ = _iter_end_

        if _iter_start_ == _iter_end_: 
            raise StopIteration

        try:
            return (
                _iter_end_ - _iter_start_,
                _iter_start_,
                _iter_end_,
                self._time_,
                # self._msg_,
                self._msg_id_,
                self._embed_
            )
        
        except:
            # self.log.warn(f" -- StopIteration")
            raise StopIteration

    def __getitem__(self, index):
        try:
            return (
                self._time_[index],
                # self._msg_[index],
                self._msg_id_[index],
                self._embed_[index]
            )
        except:
            raise IndexError