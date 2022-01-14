import os
import re

from pathlib import Path

from ..prefix import NASReposPath
from ..utils import logger


class NASRepos(object):
    
    def __init__(self, master, uri=None, branch=None, local=None):
        self.master = master
        self.uri = uri
        self.branch = branch
        self.local = local
        self.entry = []
        self.status = []
        
    def connect(self):
        # self.local.parent.mkdir(parents=True, exist_ok=True)
        self.local.mkdir(parents=True, exist_ok=True)
        if not self.local.exists():
            raise Exception(f"Not exists directory -> {self.local}")

    def exists(self, file):
        for i, path in enumerate(self.entry):
            if re.search(str(file)+'$', str(path)):
                return True
        return False
    
    def add(self, file):
        gpath = NASReposPath(self.master, local=self.local, uri=file)
        # gpath.target().parent.mkdir(parents=True, exist_ok=True)
        self.entry.append(gpath.target())
        # self.status.append(self._get_status_(gpath.target()))
        self.status.append(None)
    
    def _get_status_(self, path):
        try:
            return os.stat(path)
        except:
            return None
        
    def _modify_status_(self, src, dst):
        if not dst is None and src is None:
            return True
        if dst is None or src is None:
            return False
        return src.st_mtime != dst.st_mtime
        
    def get(self, file):
        for i, path in enumerate(self.entry):
            if re.search(str(file)+'$', str(path)):
                status = self._get_status_(path)
                update = self._modify_status_(self.status[i], status)
                return update, path, (i, status)
    
    def put(self, func, obj, file):
        for i, path in enumerate(self.entry):
            if re.search(str(file)+'$', str(path)):
                path = Path(self.local).joinpath(path)
                func(obj, path)
        
    def update_status(self, st):
        self.status[st[0]] = st[1]

    def pull(self):
        pass