import os
import re
import git
from pathlib import Path

from ..utils import logger, encrypt, shutil as mutil
from ..prefix import GitReposPath

class GitRepos(object):
    
    def __init__(self, master, uri, branch, local):
        self.master = master
        self.uri = uri
        self.branch = branch
        self.local = local
        self.entry = []
        self.status = []

    def clone(self):
        self.local.parent.mkdir(parents=True, exist_ok=True)
        self.repo = git.Repo.clone_from(
                self.uri,
                self.local,
                branch=self.branch,
                no_checkout=False)
        
    def connect(self):
        try:
            self.repo = git.Repo(self.local)
        except:
            self.clone()
            
    def add(self, file):
        gpath =  GitReposPath(self.master, local=self.local, uri=file)
        self.entry.append(gpath.target())
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
        self.pull()
        for i, path in enumerate(self.entry):
            if re.search(str(file), str(path)):
                status = self._get_status_(path)
                update = self._modify_status_(self.status[i], status)
                return update, path, (i, status)
            
    def put(self, func, obj, file):
        for i, path in enumerate(self.entry): 
            if re.search(str(file), str(path)):
                path = Path(self.local).joinpath(path)
                msg = func(obj, path)
                self.commit(path, msg)
                self.push()
                
    def update_status(self, st):
        self.status[st[0]] = st[1]
            
    def fetch(self):
        raise NotImplementedError
    
    def pull(self):
        origin = self.repo.remote(name='origin')
        origin.pull()
                
    def push(self):
        origin = self.repo.remote(name='origin')
        origin.push()
    
    def commit(self, path, msg=None):
        if msg is None:
            msg = path.name

        self.repo.git.add(path)  # to add all the working files.
        self.repo.git.commit(message=msg)
        