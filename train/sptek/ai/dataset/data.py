import numpy as np
import pandas as pd
from numpy.lib.arraysetops import isin
from ..utils import logger

class TFData(object):
    def __init__(self, master, meta, key='DATE'):
        self.log = logger.get(self)
        self._data_ = None
        self.master = master
        self.key = key
        self.meta = meta
        
    def to_datetime(self, col, format="%Y-%m-%d %H:%M:%S"):
        self._data_[col] = pd.to_datetime(self._data_[col], format=format)
        
    def to_number(self, col, astype='float64'):
        cols = {}
        for a in col:
            if a in self._data_.columns:
                cols[a] = astype
        self._data_ = self._data_.astype(cols)
        
    def to_percent(self):
        if hasattr(self.meta, 'percent'):
            items = self.meta.percent
            if not isinstance(items, list):
                items = [items]
            for item in items:
                self._data_[item.numerator] = np.round(self._data_[item.numerator] / self._data_[item.denominator] * 100)
        # logger.debug(f"\n{self._data_}")
        
    def append(self, col, value, axis=1):
        index = self._data_.index
        series = pd.Series(value, name=col, index=index)
        self._data_ = pd.concat([self._data_, series], axis=axis)
        
    def update(self, value, feature):
        if not isinstance(feature, list):
            feature = [feature]
        df = pd.DataFrame(value, columns=feature)
        self.combine_first(df)
        # logger.debug(f"\n{self._data_}")
        
    def copy(self):
        obj = TFData(self.master, self.meta, self.key)
        obj._data_ = self._data_.copy()
        return obj
    
    def from_numpy(self, data):
        if len(data.shape) > 2:
            data = data.reshape(data.shape[1], data.shape[2])
        self._data_ = pd.DataFrame(data, columns=self.meta.output)
        
    def __getitem__(self, index):
        return self._data_[index]
    
    def __repr__(self):
        return repr(self._data_)
    
    def __len__(self):
        return len(self._data_)
    
    def to_numpy(self):
        return self._data_.to_numpy()
        
    def combine_first(self, data):
        self._data_.drop(columns=data.columns, inplace=True)
        df_inner = self._data_.merge(data, how='inner', left_index=True, right_index=True)
        self._data_ = df_inner
    
    def reset_index(self):
        self._data_ = self._data_.reset_index(drop=True)
        
    def empty(self):
        return self._data_.empty
    
    @property
    def date(self):
        return self._data_[self.key]
    
    @property
    def columns(self):
        return self._data_.columns
        
    @property
    def dtypes(self):
        return self._data_.dtypes
    
    @property
    def values(self):
        return self._data_.values
    
    @property
    def index(self):
        return self._data_.index
