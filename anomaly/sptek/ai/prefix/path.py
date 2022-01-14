import re
from pathlib import Path
from datetime import datetime

from ..utils import logger, encrypt

class ModelPath(object):
    
    def __init__(self, master, parser):
        self.master = master
        self.parser = parser
        
        service = master.service('name')
        database = master.database(dtype=parser.model())
        
        meta = master.import_model(dtype='sequential', lookup="match")
        if meta is not None:
            model = meta
            path = meta.uri
            
        meta = master.machine(dtype='sequential', lookup="match")
        if meta is not None:
            model = meta
            path = meta.export
        
        if re.search('{{datetime}}', path):
            path = path.replace('{{datetime}}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
        if re.search('{{service.name}}', path):
            path = path.replace('{{service.name}}', service)
            
        if re.search('{{database.name}}', path):
            path = path.replace('{{database.name}}', database.name)
            
        if re.search('{{model.name}}', path):
            path = path.replace('{{model.name}}', model.name)
            
        if re.search('{{model.algorithm}}', path):
            path = path.replace('{{model.algorithm}}', model.algorithm)
            
        if re.search('{{database.type}}', path):
            path = path.replace('{{database.type}}', parser.model())

        self._path_ = path
        self._algorithm_ = model.algorithm
            
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
            dest = re.split('/', str(uri))[3:]
            self._target_ = self._local_.joinpath(*dest)
            
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
            dest = re.split('/', str(uri))[3:]
            self._target_ = self._local_.joinpath(*dest)
            
            
    def uri(self):
        return self._uri_
    
    def local(self):
        return self._local_
    
    def target(self):
        return self._target_
    
    def name(self):
        return self.local().name


class ScalerPath(object):
    
    def __init__(self, master):
        self.master = master
        self._path_ = None
        self._algorithm_ = None
        
        meta = master.import_model(dtype='scaling', lookup="match")
        path = meta.uri
        algorithm = meta.algorithm
        
        self._path_ = path
        self._algorithm_ = algorithm
        
    def path(self):
        return self._path_
    
    def algorithm(self):
        return self._algorithm_