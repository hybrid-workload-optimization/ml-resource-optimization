from sklearn import preprocessing
from .scaler import Scaler
from .. import prefix
from ..serving import ModelServing
from ..serving import ExportAdaptor
from ..utils import logger
import pandas as pd

class PowerScaler(Scaler):
    
    def __init__(self, master):
        super().__init__(master, preprocessing.PowerTransformer())
        self.log = logger.get(self)
        
    def export(self, path):
        try:  
            path = prefix.model.name(path)
            serving = ModelServing()
            adaptor = ExportAdaptor('scaler', serving)
            adaptor.put({
                    "model": self._scaler_,
                    "msg": 'log'
                    }, path)
        except Exception as e:
            self.log.warn(f"\n{e}")
        
    def fit(self, dataset):
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
