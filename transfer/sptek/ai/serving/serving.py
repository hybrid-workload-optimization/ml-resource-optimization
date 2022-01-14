import tensorflow as tf
import time
from pathlib import Path

from ..utils import MetaclassSingleton
from ..utils import ThreadPool


class ModelServing(metaclass=MetaclassSingleton):
    
    def __init__(self):
        self.serving = {}
        self.repos = {}
        self.session = []
        self.adaptors = []
    
    def start(self):
        for _, values in self.repos.items():
            for _, item in values.items():
                item.connect()
        # self.pull()
        pool = ThreadPool()
        pool.add_task_nowait(self._pull_service_)
    
    def attach(self, name, dest):
        if not name in self.serving.keys():
            self.serving[name] = dest[0]
        else:
            for sub, item in dest[0].items():
                self.serving[name][sub].extend(item)
            
        for key, values in self.serving.items():
            for _, item in values.items():
                self.adaptors.extend(item)
                
        self.repos[name] = dest[1] 
    
    def get(self, module, name, file):
        for adaptor in self.serving[module][name]:
            if adaptor.file() == file:
                return adaptor.get(file)
    
    def put(self, module, name, obj, file):
        for adaptor in self.serving[module][name]:
            if adaptor.file() == file:
                return adaptor.put(obj)
    
    def pull(self):
        for _, value in self.repos.items(): 
            for _, item in value.items(): 
                item.pull()
        
        for item in self.adaptors:
            if item.is_update():
                item.update()
    
    def _pull_service_(self, session=None):
        while True:            
            self.pull()            
            time.sleep(10)
            
    def filename(self, module, name):
        adaptor = self.serving[module][name][self._index_]
        return adaptor.file()
     
    def first(self, module, name):
        try:
            self._index_ = 0
            adaptor = self.serving[module][name][self._index_]
            return adaptor.get(adaptor.file())
        except:
            return None
                    
    def next(self, module, name):
        try:
            self._index_ += 1
            _start_ = self._index_
            adaptors = self.serving[module][name]
            for x in range(_start_, len(adaptors)):
                y = adaptors[x]
                self._index_ = x
                yield y.get(y.file())
        except Exception as e:
            raise StopIteration
    

class ImportAdaptor(object):
    
    def __init__(self, name, serving):
        self.name = name
        self.serving = serving
        
    def first(self):
        model = self.serving.first('import', self.name)
        filename = self.serving.filename('import', self.name)
        return Path(filename).name, model
    
    def next(self):
        for model in self.serving.next('import', self.name):
            filename = self.serving.filename('import', self.name)
            yield Path(filename).name, model

    def get(self, file):
        return self.serving.get("import", self.name, file)
    
    def put(self, module, name, obj, file):
        return self.serving.put("import", self.name, obj, file)
    
class ExportAdaptor(object):
    
    def __init__(self, name, serving):
        self.name = name
        self.serving = serving
        
    def first(self):
        model = self.serving.first('export', self.name)
        filename = self.serving.filename('export', self.name)
        return Path(filename).name, model
    
    def next(self):
        for model in self.serving.next('export', self.name):
            filename = self.serving.filename('export', self.name)
            yield Path(filename).name, model

    def get(self, file):
        return self.serving.get("export", self.name, file)
    
    def put(self, obj, file):
        return self.serving.put("export", self.name, obj, file)
    