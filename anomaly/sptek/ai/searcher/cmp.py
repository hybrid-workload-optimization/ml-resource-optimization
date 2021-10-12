import json

from ..utils import logger, serialize
from ..rest import Requester

class CMPSearcher(object):
    
    def __init__(self, meta):
        self.log = logger.get(self)
        self.db = meta
        self._data_ = None
        self._iter_index_ = 0
        

        # build header meta
        self._head_key_ = [
            "searchTime",
            "logCount",
            "is_VALID",
            "invalid_MSG"
        ]
        

    def _get_head_(self, data):
        meta = {}
        for key in self._head_key_:
            meta[key] = data[key]
        return meta


    def _search_(self, log_type, start_time, end_time, message):
        self.log.debug(f"{log_type}, {start_time}, {end_time}, {message}")
        requester = Requester(self.db.uri)
        return requester.request({
            "log_type":log_type,
            "start_log_time": start_time,
            "end_log_time": end_time,
            "message": message
        })

    def _search_not_message_(self, log_type, start_time, end_time):
        self.log.debug(f"{log_type}, {start_time}, {end_time}")
        requester = Requester(self.db.uri)
        return requester.request({
            "log_type" : log_type,
            "start_log_time" : start_time,
            "end_log_time" : end_time
        })

    def search(self, start_time, end_time, message=None):
        self.log.info(f"Collect logs from {start_time} to {end_time} -- filter message: {message}")
        if message:
            self._data_ = self._search_(
                self.db.log.type,
                start_time,
                end_time,
                message
            )
            self._head_ = self._get_head_(self._data_)

        else:
            self._data_ = self._search_not_message_(
                self.db.log_type,
                start_time,
                end_time
            )
            self._head_ = self._get_head_(self._data_)
        
        return self._data_

    def dumps(self, data=None):
        if data is None:
            data = self._data_

        dump = json.dumps(
            data,
            default=serialize.json_serialize,
            indent=4)
        logger.debug(f"{dump}")

    def count(self):
        return self._head_['logCount']

    def _take_(self, index):
        return self._data_['logCmpList'][index]

    def take(self, index):
        index = index - 1
        if index < 0:
            return None

        if index < self.count():
            return self._take_(index)

        return None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_index_ >= self.count():
            raise StopIteration

        self._iter_index_ = self._iter_index_ + 1
        return self.take(self._iter_index_)