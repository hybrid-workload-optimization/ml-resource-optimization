import copy
import numpy as np
import pandas as pd
from sklearn import preprocessing

from .. import prefix
from ..utils import logger
from ..serving import ModelServing


class StandardScaler(object):
    
    def __init__(self, master, meta):
        self.log = logger.get(self)
        self.master = master
        self.meta = meta
        self._scaler_ = preprocessing.StandardScaler()
    
    def load(self, path=None):
        try:
            if path is None:
                path = self.meta.uri
            serving = ModelServing()
            self._scaler_ = serving.get('scaler', prefix.model.name(str(path)))
        except Exception as e:
            self.log.error(f"Not import scaler model completed.\n{e}")
    
    def fit(self, X):
        self._scaler_.fit(X.dataset)
    
    def transform(self, X):
        scaled = self._scaler_.transform(X.dataset)
        X.dataset = pd.DataFrame(scaled, columns=X.dataset.columns)
        return X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X):
        return self._scaler_.inverse_transform(X)
        
    def export(self, path):
        try:
            serving = ModelServing()
            serving.put(
                'scaler', {
                    "model": self._scaler_,
                    "msg": 'z-score'
                    },
                prefix.model.name(path)
            )            
        except Exception as e:
            self.log.warn(f"{e}")