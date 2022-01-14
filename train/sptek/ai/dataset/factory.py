from .data import TFData
from .searcher import DataSearcher
from .testset import TestSet
from .trainset import TrainSet
from .. import prefix
from ..serving import ImportAdaptor
from ..serving import ModelServing
from ..utils import logger


class DataFactory(object):

    def __init__(self):
        pass

    @classmethod
    def load(cls, path):
        file = prefix.model.name(path)
        adaptor = ImportAdaptor('database', ModelServing())
        return adaptor.get(file)
    
    @classmethod
    def build(cls, master, dataset, action='output'):
        meta = master.machine(dtype='features')
        obj = TFData(master, meta, key=meta.key)
        obj.from_numpy(dataset)
        return obj
    
    
class SplitterFactory(object):
    
    @classmethod
    def create(cls, master, dataset, action="train"):
   
        if action == 'train':
            return TrainSet(master, dataset)
        
        if action == 'forecast':
            return TestSet(master, dataset)
        
        raise Exception(f"Unknown the value of action variable. (action={action})")
    
    
class SearcherFactory(object):
    @classmethod
    def create(cls, master, dataset, action="train"):
        meta = master.target(dtype='resource')
        # logger.debug(f"{meta}")
        
        return DataSearcher(master, dataset, res=meta, action=action)
        