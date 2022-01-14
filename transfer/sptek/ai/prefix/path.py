import re
from pathlib import Path
from datetime import datetime

from . import repos
from . import model
from ..utils import encrypt
from ..utils import logger


def realpath(path, suffix):
    
    keys = []
    if isinstance(suffix, dict):
        keys = suffix.keys()
    
    real = []
    if 'target' in keys:
        arr = suffix['target']
        if not isinstance(arr, list):
            arr = [arr]
        for item in arr:
            real.append(str(path).replace('{{target}}', item))
            
    if len(real) == 0:
        real.append(path)
            
    return list(set(real))


def get_target_path(meta):
    _name_ = repos.name(meta.uri)
    _path_ = model.name(meta.uri)
    
    if hasattr(meta, 'target'):
        _path_ = realpath(_path_, {'target': meta.target} if hasattr(meta, 'target') else None )
            
    if hasattr(meta, 'legacy-target'):     
        _path_ = realpath(_path_, {'target': meta['legacy-target']} if hasattr(meta, 'legacy-target') else None )

    if not isinstance(_path_, list):
        _path_ = [_path_]
    
    return _name_, _path_


class ModelPath(object):
    
    def __init__(self, master, path=None):
        self.master = master
        self._path_ = path
        
        meta = master.import_model(dtype='sequential', lookup="match")
        if meta:
            _uri_ = meta.uri
            _algorithm_ = meta.algorithm
        
        meta = master.export_model(dtype='sequential', lookup="match")
        if meta:
            _uri_ = meta.uri
            _algorithm_ = meta.algorithm
            
        if re.search('{{target.value}}', str(_uri_)):
            _path_ = str(_uri_).replace('{{target.value}}', path)
        else:
            _path_ = str(_uri_)
            
        self._path_ = _path_
        self._algorithm_ = _algorithm_
    
    def path(self):
        return self._path_
    
    def algorithm(self):
        return self._algorithm_
    
    
class ClusterPath(object):
    
    def __init__(self, master):
        self.master = master
        self._path_ = None
        self._algorithm_ = None
        
        meta = master.import_model(dtype='cluster', lookup="match")
        if meta is not None:
            path = meta.uri
            algorithm = meta.algorithm
            
        meta = master.machine(dtype='cluster', lookup="match")
        if meta is not None:
            path = meta.export
            algorithm = meta.algorithm

        self._path_ = path
        self._algorithm_ = algorithm
        
    def path(self):
        return self._path_
    
    def algorithm(self):
        return self._algorithm_
    
    
class GitReposPath(object):
    
    def __init__(self, master, meta=None, uri=None, local=None):
        self.key = 'b3940b42a28898a7b3673e8add766202ca38ad805f7cdb3ed9cdba56f14580d9'
        self.master = master
        self._dir_ = None
        self._uri_ = None
        self._local_ = None
        self._target_ = None
        
        if hasattr(master, 'work_dir'):
            self._dir_ = self.master.work_dir
        
        if meta is not None:
            self._uri_ = meta.uri
            self.user = meta.user
            self.pwd = encrypt.decrypt(meta.pwd, self.key).decode('utf-8')
            
            if re.search('{{user}}', self._uri_):
                self._uri_ = self._uri_.replace('{{user}}', self.user)
                
            if re.search('{{password}}', self._uri_):
                self._uri_ = self._uri_.replace('{{password}}', self.pwd)
            
            if local is None:
                local = meta.local

        if local is not None:
            self._local_ = Path(self._dir_).joinpath(Path(local))
        else:
            self._local_ = local
        
        if uri is not None:            
            if str(uri)[0] == '/':
                dest = re.split('/', str(uri))[3:]
                self._target_ = self._local_.joinpath(*dest)
            else:
                self._target_ = self._local_.joinpath(uri)
            
            
    def uri(self):
        return self._uri_
    
    def local(self):
        return self._local_
    
    def target(self):
        return self._target_
    
    def branch(self):
        return 'master'
    
    
class NASReposPath(object):
    
    def __init__(self, master, meta=None, uri=None, local=None):
        self.master = master
        self._dir_ = None
        self._uri_ = None
        self._local_ = None
        self._target_ = None
        
        if hasattr(master, 'work_dir'):
            self._dir_ = self.master.work_dir
        
        if local is None:
            local = meta.local
            
        if local is None:
            self._local_ = Path(self._dir_)
        else:
            self._local_ = Path(self._dir_).joinpath(Path(local))

        if uri is not None:            
            if str(uri)[0] == '/':
                dest = re.split('/', str(uri))[3:]
                self._target_ = self._local_.joinpath(*dest)
            else:
                self._target_ = self._local_.joinpath(uri)
            
            
    def uri(self):
        return self._uri_
    
    def local(self):
        return self._local_
    
    def target(self):
        return self._target_
    
    def name(self):
        return self.local().name


class ScalerPath(object):
    
    def __init__(self, master, path=None):
        self.master = master
        self._path_ = None
        self._algorithm_ = None
        
        meta = master.import_model(dtype='scaler', lookup="match")
        if meta:
            _uri_ = meta.uri
            _algorithm_ = meta.algorithm
        
        meta = master.export_model(dtype='scaler', lookup="match")
        if meta:
            _uri_ = meta.uri
            _algorithm_ = meta.algorithm
            
        if path and re.search('{{target.value}}', str(_uri_)):
            _path_ = str(_uri_).replace('{{target.value}}', path)
        else:
            _path_ = str(_uri_)
            
        self._path_ = _path_
        self._algorithm_ = _algorithm_
        
    def path(self):
        return self._path_
    
    def algorithm(self):
        return self._algorithm_
    

class DataPath(object):
    def __init__(self, master=None, path=None):
        self.master = master
        self._path_ = path
        self.__call__(master, path)
        
    def __call__(self, master, path=None):
        if master is None:
            return self
        
        self.master = master
        self._path_ = path 
        
        if path is None:
            meta = master.database()
            self._path_ = meta.uri
        
        self.size = 1
        return self
        
    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index = self.index + 1
        if self.index < self.size:
            return self._path_
        raise StopIteration
        
    def path(self):
        return self._path_
    

class TargetPath(object):
    
    def __init__(self, master=None, path=None):
        self.master = master
        self._path_ = path
        self.__call__(master, path)

    def __call__(self, master, path):
        if master is None or path is None:
            return self
        
        self.master = master
        self._path_ = path
        self.values = []
        
        if re.search('{{target.value}}', str(path)):
            meta = master.target(dtype="resource", lookup="match")
            for value in  meta.value:
                self.values.append(str(path).replace('{{target.value}}', value))
        else:
            self.values.append(str(path))
                
        self.size = len(self.values)
        return self
            
    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index = self.index + 1
        if self.index < self.size:
            return self.values[self.index]
        raise StopIteration
    
    def path(self):
        if len(self.values):
            return self.values[0]
    