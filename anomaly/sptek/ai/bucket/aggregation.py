import time
import numpy as np
import pandas as pd
from ..utils import logger, ThreadPool, thread, stamp
from ..scheduler import JobTrigger

import threading
locker = threading.Lock()

class TermAggregation(JobTrigger):

    def __init__(self, master, cluster, searcher):
        self.log = logger.get(self)
        self.master = master
        self.cluster = cluster
        self._searcher_ = searcher
        self.session = []
        self._session_completed_ = False
        self.bucket_vector = BucketVector(self.cluster.n_clusters, searcher.scroll)
        self._shape_ = 0
        self._time_ = []        

    def count(self):
        return self._shape_    

    def shape(self):
        return (self.count(), 3)
    
    def current(self):
        return self._searcher_.current()
    
    def ago(self):
        return self._searcher_.ago()
    
    def interval(self):
        return self._searcher_.interval()
    
    def bucket_series(self):
        return self.bucket_vector

    def fit(self, id=0):
   
        for _start_, _end_, _size_, _index_, _time_, _msg_, _embed_ in self.searcher():
            self._shape_ = self._shape_ + _size_
            self._time_.extend(_time_)
            bucket = Bucket(self.cluster.n_clusters, _start_, _end_)
            
            if _size_ == 0:
                term = Term(_start_, _end_, None, None, None, None)
                bucket.append(-1, term)
            
            if _size_ == 1:
                label = self.cluster.predict([_embed_])
                term = Term(_start_, _end_, _index_[0], None, None, None)
                bucket.append(label[0], term)
                    
            if _size_ > 1:
                predict = self.cluster.predict(_embed_)
                for i, label in enumerate(predict):
                    term = Term(_start_, _end_, _index_[i], None, None, None)
                    bucket.append(label, term)
            
            # self.log.debug(f"bucket term count: {bucket.term_count()} -- (id={id})")
            self.bucket_vector.append(bucket)
            # self._bucket_.append(self._bucket_index_)
            # self._bucket_index_ = self._bucket_index_ + 1
            # self.log.debug(f"append bucket: {self.shape()} -- (id={id})")
            
        # self.log.debug(f"datasearcher recv count = {self.searcher().count()}")
        # self.log.debug(f"----------- _iter_index_ = {None}")
        # self.log.debug(f"----------- _iter_start_ = {self.searcher()._iter_start_}")
        # self.log.debug(f"----------- _iter_end_ = {self.searcher()._iter_end_}")
        
        _times_ = pd.to_datetime(pd.Series(self._time_))
        self.log.debug(f"bucket _size_ = {self.count()}")
        self.log.debug(f"------- _start_ = {None}")
        self.log.debug(f"------- _end_ = {None}")
        self.log.debug(f"------- _time_ size = {_times_.shape}")
        self.log.debug(f"------- until time = {_times_.min()} ~ {_times_.max()}")
        

    def fork_fit(self, fork=1):
        pool = ThreadPool()
        for _ in range(fork):
            self.session.append(pool.add_task(self._fork_serve_))
        pool.add_task(self._check_fork_)

    def _fork_serve_(self, session=None):
        # self.log.info(f"fork a subprocess for category aggregation (term). -- (id={session})")
        self.fit(session)

    def _check_fork_(self, session=None):
        start = time.time()
        self._wait_completed_()
        duration =  time.time() - start        
        self.log.info(f"all subprocess for category aggregation (term) completed.-- {self.shape()} -- (duration={duration:.5f}s)")
        self._session_completed_ = True

    def _wait_completed_(self):
        with locker:
            pool = ThreadPool()
            _del_ = []
            for s in self.session:
                pool.wait_completion(s)
                _del_.append(s)
            # self.session = []
            for d in _del_:
                self.session.remove(d)
        
    def wait_completed(self):
        while not self._session_completed_:
            thread.nsleep(1)

    def searcher(self):
        return self._searcher_
    
    def job(self):
        
        start = time.time()
        
        inputs = self.count()
        self.log.debug(f"job -- bucket -- (shape={self.shape()})")
        
        pool = ThreadPool()
        self.session.append(pool.add_task(self._fork_serve_))
        self._wait_completed_()
        
        inputs = self.count() - inputs
                
        # check out old data.
        _ago_ = stamp.prev_time(self.searcher().current(), self.searcher().ago())
        _times_ = pd.to_datetime(pd.Series(self._time_))
        # _times_ = _times_.sort_values(ascending=True)
        _filter_ = _times_ <= _ago_
        values = _times_[_filter_]
        indexs = _times_.index[_filter_].tolist()
        indexs.sort(reverse=True)
        
        self.log.debug(f"min={pd.Timestamp(values.min())}, max={pd.Timestamp(values.max())}")
        
        _del_t_str_ = 'no match'
        remove_size = len(indexs)
        if remove_size:
            self._shape_ = self._shape_ - remove_size
            _del_times_ = [ self._time_[x] for x in indexs ]
            _del_times_ = pd.to_datetime(pd.Series(_del_times_))
            _del_t_str_ = f"{_del_times_.min()} ~ {_del_times_.max()}"
            del _del_times_
    
        for i, bucket in self.bucket_vector:
            if bucket.compare(pd.Timestamp(values.min()), pd.Timestamp(values.max())):
                self.bucket_vector.delete(i)
                break
            
        # remove element
        for d in indexs:
            del self._time_[d]
                    
        duration =  time.time() - start
        self.log.debug(f"job -- bucket -- until time = {_times_.min()} ~ {_times_.max()}")
        self.log.info(f"job -- bucket -- input={inputs}, remove={remove_size}, shape={self.shape()} -- bucket={self.bucket_vector.count()} -- search= ~{self.current()} -- remove= ~{_ago_} -- (duration={duration:.5f}s)")



    
class Term(object):

    def __init__(self, start, end, index, stamp, message, dense):
        self.start = start
        self.end = end
        self.index = index
        self._time_ = stamp
        self._msg_ = message
        self._dense_ = dense        

    def __repr__(self):
        return '\n'.join([
            f"{super().__repr__()}",
            f"index : {self.index}",
            # f"time : {text.truncate_middle(str(self._time_.values[0]), 60)}",
            # f"message : {text.truncate_middle(str(self._msg_.values[0]), 60)}",
            # f"dense : {text.truncate_middle(str(self._dense_.values[0]), 60)}"
        ])


class TermVector(object):
    def __init__(self):
        self.vector = []

    def append(self, term):
        if isinstance(term, Term):
            self.vector.append(term)

    def count(self):
        return len(self.vector)

    def __len__(self):
        return len(self.vector)

class Category(object):
    
    def __init__(self, label):
        self.label = label
        self.term_vector = TermVector()

    def append(self, term):
        if isinstance(term, Term):
            self.term_vector.append(term)

    def count(self):
        return self.term_vector.count()

    def __repr__(self):
        return '\n'.join([
            f"{super().__repr__()}",
            f"label : {self.label}",
            f"term count : {self.count()}"
        ])
        

class Bucket(object):
    def __init__(self, n_clusters, start, end):
        self.n_clusters = n_clusters
        self._start_ = start
        self._end_ = end
        self.categories = []
        for label in range(n_clusters):
            self.categories.append(Category(label))

    def append(self, label, term):
        if isinstance(term, Term):
            self.categories[label].append(term)

    def term_count(self, action='all'):
        count = 0
        for v in self.categories:
            count = count + v.count()
        return count
    
    def compare(self, start, end):
        if isinstance(self._start_, str):
            self._start_ = stamp.strptime(self._start_)
            
        if isinstance(self._end_, str):
            self._end_ = stamp.strptime(self._end_)
            
        if isinstance(start, str):
            start = stamp.strptime(start)
            
        if isinstance(end, str):
            end = stamp.strptime(end)
        
        return self._start_ == start and self._end_ == end
        

    def __repr__(self):
        head = [
            f"{super().__repr__()}",
            f"* category size : {self.n_clusters}",
            f"* term count : {self.term_count()}",
            f"* detail:"
        ]

        detail = []
        for v in self.category_vector:
            v_str = "-- category: {0}\n-- terms: {1}".format(v.label, v.count())
            detail.append(v_str)

        body = []
        body.extend(head)
        body.extend(detail)
        return '\n'.join(body)
            

class BucketVector(object):
    def __init__(self, n_cluster, scroll):
        self.n_cluster = n_cluster
        self.scroll = scroll
        self.vector = []
        self._size_ = 0
        self._index_ = 0

    def append(self, bucket):
        if isinstance(bucket, Bucket):
            self.vector.append(bucket)
            self._size_ = self._size_ + 1
            
    def delete(self, index):
        del self.vector[index]
        self._size_ = self._size_ - 1
        
    def count(self):
        # return len(self.vector)
        return self._size_
    
    def __getitem__(self, index):
        return self.vector[index]

    def __repr__(self):
        return '\n'.join([
            f"{super().__repr__()}",
            f"* bucket size : {len(self.vector)}",
        ])
        
    def __iter__(self):
        self._index_ = 0
        return self

    def __next__(self):
        if self._index_ < self._size_:
            _index_ = self._index_
            self._index_ = self._index_ + 1
            return _index_, self.vector[_index_]
        else:
            raise StopIteration