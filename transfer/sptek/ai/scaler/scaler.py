from .. import prefix
from ..serving import ModelServing
from ..serving import ImportAdaptor
from ..utils import logger

class Scaler(object):
        
    def __init__(self, master, scaler):
        self.log = logger.get(self)
        self.master = master
        self._scaler_ = scaler
        self._empty_ = True
    
    def load(self, path):
        file = prefix.model.name(path)
        adaptor = ImportAdaptor('scaler', ModelServing())
        self._scaler_ = adaptor.get(file)
        if self._scaler_ :
            self._empty_ = False
    
    def fit(self, dataset):
        raise NotImplementedError
    
    def transform(self, dataset):
        raise NotImplementedError
    
    def inverse_transform(self, dataset):
        raise NotImplementedError
    
    def empty(self):
        return self._empty_
