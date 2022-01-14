import string
import pandas as pd

from ..utils import logger

class CMPDataSet(object):
    def __init__(self, searcher):
        self.log = logger.get(self)
        self._searcher_ = searcher
        self._dataset_ = None
        self._start_time_ = None
        self._end_time = None
        

    def build(self):
        # transelate json to matrix
        # df = pd.DataFrame.from_dict({'time': [0], 'message': [0]})
        df = pd.DataFrame()
        
        for index, record in enumerate(self._searcher_):
            # removing punctuation and splitting on spaces.    
            # remove punctuations and digits from oldtext
            oldtext = record['message']
            exclist = string.punctuation + string.digits    
            table_ = str.maketrans(exclist, ' '*len(exclist))
            newtext = ' '.join(oldtext.translate(table_).split())
            
            df = df.append({
                    'time': record['log_time'],
                    'message': newtext
                    },
                ignore_index=True)
        
        self._dataset_ = df
        # self._dataset_ = df.set_index('time', drop=True)
        return self._dataset_

    def data(self):
        return self._dataset_

    def filter(self, start, end):
        self._start_time_ = start
        self._end_time = end
        self._searcher_.search(start, end)

    def searcher(self):
        return self._searcher_

    def messages(self):
        return self.data()['message']

    def append(self, ds):
        df = pd.DataFrame.from_dict(ds)
        self._dataset_ = pd.concat([self._dataset_, df], axis=1)
        
    def copy(self):
        clone = CMPDataSet(self._searcher_)
        clone._dataset_ = self._dataset_.copy()
        clone._start_time_ = self._start_time_
        clone._end_time = self._end_time
        return clone

    def shape(self):
        return self._dataset_.shape

    def sort(self):
        self._dataset_ = self._dataset_.sort_index()

    def drop_duplicates(self):
        self._dataset_ = self._dataset_.groupby(level=0).last()

    def set_index(self):
        self._dataset_ = self._dataset_.set_index('time', drop=True)