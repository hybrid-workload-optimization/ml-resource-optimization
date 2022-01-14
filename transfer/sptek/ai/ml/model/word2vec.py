import numpy as np
import pandas as pd
import tensorflow as tf
import threading
import time

from operator import itemgetter
from scipy.spatial import distance

from ... import prefix
from ...serving import ModelServing
from ...utils import logger
from ...utils import path
from ...utils import thread
from ...utils import ThreadPool

locker = threading.Lock()

class EmbeddingVector(object):
    
    def __init__(self, meta, searcher=None):
        self.log = logger.get(self)
        self.meta = meta
        self._searcher_ = searcher
        self._embed_set_ = []
        self.path = path.uri_to_absolutized(meta.uri)
        self.session = []
        self._session_completed_ = False 
        
        self._time_ = []
        self._msg_ = []
        self._embed_ = []
        self._shape_ = 0

    def load(self):
        try:
            serving = ModelServing()
            self.model = serving.get('keras', prefix.model.name(str(self.meta.uri)))
        except Exception as e:
            self.log.error(f"Not import embedding model completed. -- (tf)\n{e}")

    def dataset(self, searcher):
        self._searcher_ = searcher

    def searcher(self):
        return self._searcher_

    def count(self):
        return self._shape_
    
    def shape(self):
        return (self.count(), 3)

    def _array_to_embedding_(self, message):
        return self.model(message)

    def build(self, id = 0):
        for _size_, _start_, _end_, _time_, _msg_, _embed_ in self.searcher():
            _temp_ = self._array_to_embedding_(_embed_[_start_:_end_])
            _embed_ = _temp_.numpy().tolist()
            self._time_.extend(_time_[_start_:_end_])
            self._msg_.extend(_msg_[_start_:_end_])
            self._embed_.extend(_embed_)
            self._shape_ = self._shape_ + _size_

    def fork(self, fork=1):
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
        self.log.info(f"all subprocess for embedding data to dense vector completed. -- {self.shape()} -- (duration= {duration:.5f}s)")
        self._session_completed_ = True
    
    def _wait_completed_(self):
        pool = ThreadPool()
        for s in self.session:
            pool.wait_completion(s)
        self.session = []

    def wait_completed(self):
        while not self._session_completed_:
            thread.nsleep(1)

    def __repr__(self):
        return repr(self._embed_set_)

    def __getitem__(self, index):
        try:
            return (
                itemgetter(*index)(self._time_),
                itemgetter(*index)(self._msg_),
                itemgetter(*index)(self._embed_)
            )
        except:
            raise IndexError

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        current = self.index
        self.index = self.index + 1
        try:
            return self._embed_set_[current:current+1]
        except Exception as e:
            print(e)
            raise StopIteration