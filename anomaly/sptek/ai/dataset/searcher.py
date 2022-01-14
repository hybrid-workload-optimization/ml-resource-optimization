
from numpy.lib.arraysetops import isin
from ..utils import logger, thread


class DataSearcher(object):
    
    def __init__(self, dataset):
        self.log = logger.get(self)
        self._dataset_ = dataset
        # self.current = dataset.current
        # self.ago = dataset.ago
        # self.interval = dataset.interval
        self.fit()

    def fit(self):
        self._iter_current_ = 0

    def count(self):
        return self._dataset_.count()
    
    def current(self):
        return self._dataset_.current()
    
    def ago(self):
        return self._dataset_.ago()
    
    def interval(self):
        return self._dataset_.interval()
    
    def last_time(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):

        _count_ = self.count()
        while _count_ <= self._iter_current_ and not self._dataset_._session_completed_:
            thread.usleep(1)
            _count_ = self.count()
            
        self.log.debug(f"count={_count_}")

        _iter_start_ = self._iter_current_
        _iter_end_ = _count_
        self._iter_current_ = _iter_end_
        
        if _iter_start_ >= _iter_end_: 
            raise StopIteration

        try:
            return (
                _iter_end_ - _iter_start_,
                _iter_start_,
                _iter_end_,                
                self._dataset_._time_,
                self._dataset_._msg_id_,
                self._dataset_._embed_
            )
        except:
            raise StopIteration

    def __getitem__(self, index):
        try:
            if isinstance(index, str):
                if 'dense_vector' in index:
                    if hasattr(self._dataset_, '_embed_'):
                        return self._dataset_._embed_
                if 'log_time' in index:
                    if hasattr(self._dataset_, '_time_'):
                        return self._dataset_._time_
                if 'message' in index:
                    if hasattr(self._dataset_, '_msg_'):
                        return self._dataset_._msg_
                if 'msg_id' in index:
                    if hasattr(self._dataset_, '_msg_id_'):
                        return self._dataset_._msg_id_
            else:
                return self._dataset_[index]
        except:
            raise IndexError