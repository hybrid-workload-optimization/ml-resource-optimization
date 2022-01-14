import pickle
import time
import joblib
import tensorflow as tf
from ..utils import logger
from ..utils import ThreadPool
from ..utils import MetaclassSingleton


class ModelServing(metaclass=MetaclassSingleton):
    
    def __init__(self):
        self.serving = None
        self.repos = None
        self.session = []
        self.adaptors = []
    
    def start(self):
        for _, item in self.repos.items():
            item.connect()
        self.pull()
        pool = ThreadPool()
        self.session.append(pool.add_task_nowait(self._pull_service_))
    
    def attach(self, dest):
        if self.serving is not None:
            raise Exception('Not attach repository adaptors.')
        self.serving = dest[0]
        self.repos = dest[1] 
        
        for _, item in self.serving.items():
            self.adaptors.extend(item)
    
    def get(self, name, file):
        for adaptor in self.serving[name]:
            if adaptor.file() == file:
                return adaptor.get(file)
    
    def put(self, name, obj, file):
        for adaptor in self.serving[name]:
            if adaptor.file() == file:
                return adaptor.put(obj)
    
    def pull(self):
        for _, item in self.repos.items():
            item.pull()
            
        for item in self.adaptors:
            if item.is_update():
                item.update()
    
    def _pull_service_(self, session=None):
        while True:            
            self.pull()            
            time.sleep(10)
