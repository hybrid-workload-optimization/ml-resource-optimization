from datetime import datetime
import re
import json
import time
import threading
import pandas as pd
from tensorflow.python.ops.gen_array_ops import reverse

from ..utils import logger, serialize, ThreadPool, TimeGenerator, thread, stamp, MetaclassSingleton
from ..rest import Requester
from ..scheduler import JobTrigger

locker = threading.Lock()

class CMPSearcher(JobTrigger, metaclass=MetaclassSingleton):
    
    def __init__(self, master, db, meta):
        self.log = logger.get(self)
        self.master = master
        self.db = db
        self._data_ = None
        self._iter_index_ = 0
        self._fork_serve_finish = False

        self._head_ = {
            "searchTime": 0,
            "logCount": 0,
            "is_VALID": False,
            "invalid_MSG": ''
        }
        
        self._head_key_ = list(self._head_.keys())
        self._data_ = []
        self._log_time_ = []
        self._mongo_id_ = []
        self._dense_vector_ = []
        self._session_ = []
        self._session_completed_ = False
        self._searcher_ = meta
                
        self.log.info(db)
        
    def __call__(self, meta):
        # fetch meta data
        self.scroll = meta.scroll
        self._ago_ = meta.ago
        self._current_ = meta.current
        if isinstance(self._current_, str):
            self._current_ = TimeGenerator.strptime(self._current_)
        self.start = stamp.prev_time(self._current_, self._ago_)
        self.log.info(f"search logs from {self.start} until {self._current_}")
        self._interval_ = meta.interval

    def _get_head_(self, data):
        meta = {}
        for key in self._head_key_:
            meta[key] = data[key]
        return meta


    def _search_(self, log_type, start_time, end_time, message):
        requester = Requester(self.db.uri)
        return requester.request({
            "log_type":log_type,
            "start_log_time": start_time,
            "end_log_time": end_time,
            "message": message
        })

    def _search_not_message_(self, log_type, start_time, end_time):
        requester = Requester(self.db.uri)
        return requester.request({
            "log_type" : log_type,
            "start_log_time" : start_time,
            "end_log_time" : end_time
        })

    def search(self, start_time=None, end_time=None, message=None):
        if message:
            _data_ = self._search_(
                self.db.type,
                start_time,
                end_time,
                message
            )
            _head_ = self._get_head_(_data_)

        else:
            _data_ = self._search_not_message_(
                self.db.type,
                start_time,
                end_time
            )
            _head_ = self._get_head_(_data_)
        
        return _head_, _data_

    def count(self):
        return self._head_['logCount']
    
    def current(self):
        return self._current_
    
    def ago(self):
        return self._ago_
    
    def interval(self):
        return self._interval_
    
    def shape(self):
        return (self.count(), 0)

    def __iter__(self):
        return self

    def __next__(self):

        _count_ = self.count()
        while _count_ <= self._iter_index_ and not self._session_completed_:
            thread.usleep(1)
            _count_ = self.count()
            
        self.log.debug(f"count={_count_}")

        self._iter_start_ = self._iter_index_
        self._iter_end_ = _count_
        self._iter_index_ = self._iter_end_
        
        if self._iter_start_ == self._iter_end_: 
            raise StopIteration
        
        try:
            return (self._iter_end_-self._iter_start_, self._iter_start_, self._iter_end_, self._log_time_, self._mongo_id_, self._dense_vector_)
        
        except:
            raise StopIteration

    def _into_queue_(self):
        return self.__next__()

    def _append_head_(self, head):
        for key in self._head_key_:
            if key == "is_VALID":
                self._head_[key] =  head[key]
            else:
                self._head_[key] = self._head_[key] + head[key]
    
    def _append_data_(self, data):
        for item in data['logCmpList']:
            self._log_time_.append(item['log_time'])
            self._mongo_id_.append(item['mongo_id'])
            self._dense_vector_.append(serialize.to_vector(item['message_vector']))

    def searcher(self):
        return self._searcher_
        
    def _pull_(self, session=None, start=None, end=None):
        head, data = self.search(start, end)
        self._append_data_(data)
        self._append_head_(head)

    def fork(self):
        pool = ThreadPool()
        self._session_.append(pool.add_task(self._fork_serve_))
        pool.add_task(self._check_fork_)
        
    
    def _fork_serve_(self, session=None):

        if not self.scroll and not self._ago_ and not self._current_:
            self.search()

        pool = ThreadPool()
        _start_ = _end_ = self.start
        while _end_ != self._current_:
            _end_ = stamp.next_time(_end_, self.scroll)
            if _end_ > self._current_:
                _end_ = self._current_
            
            self._session_.append(pool.add_task(self._pull_, start=_start_, end=_end_))
            _start_ = stamp.next_time(_end_, '1s')

    def wait_completed(self):
        while not self._session_completed_:
            thread.nsleep(1)
        
    def _check_fork_(self, session=None):
        start = time.time()
        self._wait_completed_()
        duration =  time.time() - start
        self.log.info(f"search logs completed. -- {self.shape()} -- (duration={duration:.5f}s)")
        self._session_completed_ = True

    def _wait_completed_(self):
        with locker:
            pool = ThreadPool()
            _del_ = []
            for s in self._session_:
                pool.wait_completion(s)
                _del_.append(s)
            for d in _del_:
                self._session_.remove(d)

    def job(self, dateTime=None):
        
        start = time.time()
        
        inputs = self.count()
        self.log.debug(f"job -- logs -- (shape={self.shape()})")
            
        # get data at regular intervals.
        pool = ThreadPool()
        _start_ = stamp.next_time(self._current_, '1s') 
        if not dateTime is None:
            _end_ = stamp.strptime(dateTime)
        else:    
            _end_ = stamp.next_time(self._current_, self._interval_) 
        self._session_.append(pool.add_task(self._pull_, start=_start_, end=_end_))
        self._wait_completed_()
                
        inputs = self.count() - inputs
    
        # check out old data.
        _ago_ = stamp.prev_time(_end_, self._ago_)
        _times_ = pd.to_datetime(pd.Series(self._log_time_))
        _filter_ = _times_ <= _ago_
        indexs = _times_.index[_filter_].tolist()
        indexs.sort(reverse=True)
        
        _del_t_str_ = 'no match'
        remove_size = len(indexs)
        if remove_size:
            self._head_['logCount'] = self._head_['logCount'] - remove_size
            self._iter_index_ = self._iter_index_ - remove_size
            _del_times_ = [ self._log_time_[x] for x in indexs ]
            _del_times_ = pd.to_datetime(pd.Series(_del_times_))
            _del_t_str_ = f"{_del_times_.min()} ~ {_del_times_.max()}"
            del _del_times_
        
        # remove element
        for d in indexs:
            del self._log_time_[d]
            del self._mongo_id_[d]
        
        # update current time
        if _end_ > self._current_:
            self._current_ = _end_
        
        duration =  time.time() - start        
        self.log.debug(f"job -- logs -- until time = {_times_.min()} ~ {_times_.max()}")
        self.log.info(f"job -- logs -- input={inputs}, remove={remove_size}, shape={self.shape()} -- search= ~{self._current_} -- remove= ~{_ago_} -- (duration={duration:.5f}s)")
                