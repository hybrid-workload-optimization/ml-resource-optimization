import pandas as pd
from sklearn import preprocessing

from .. import prefix
from ..serving import ExportAdaptor
from ..serving import ImportAdaptor
from ..serving import ModelServing
from ..dataset import TFData
from ..utils import logger

class Scaler(object):
        
    def __init__(self, master, scaler):
        self.log = logger.get(self)
        self.master = master
        self._scaler_ = scaler
    
    def load(self, path):
        file = prefix.model.name(path)
        adaptor = ImportAdaptor('scaler', ModelServing())
        self._scaler_ = adaptor.get(file)
    
    def fit(self, dataset):
        raise NotImplementedError
    
    def transform(self, dataset):
        raise NotImplementedError
    
    def inverse_transform(self, dataset):
        raise NotImplementedError

class StandardScaler(Scaler):
    
    def __init__(self, master):
        super().__init__(master, preprocessing.StandardScaler())
        self.log = logger.get(self)
        
    def export(self, path):
        try:  
            path = prefix.model.name(path)
            serving = ModelServing()
            adaptor = ExportAdaptor('scaler', serving)
            adaptor.put({
                    "model": self._scaler_,
                    "msg": 'z-score'
                    }, path)
        except Exception as e:
            self.log.warn(f"\n{e}")
            
            
    
    def fit(self, dataset):
        # self._scaler_.fit(dataset)
        X = dataset[dataset.meta.input].to_numpy()
        self._scaler_.fit(X)
    
    def transform(self, dataset):
        X = dataset[dataset.meta.input].to_numpy()
        scaled = self._scaler_.transform(X)
        lookup = pd.DataFrame(scaled, columns=dataset.meta.input, index=dataset.index)
        
        obj = dataset.copy()
        obj.combine_first(lookup)
        return obj
    
    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
        
    def inverse_transform(self, dataset):
        return self._scaler_.inverse_transform(dataset)
        