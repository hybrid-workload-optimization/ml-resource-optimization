from os import removexattr
from pathlib import Path
from .. import prefix
from .git import GitRepos
from .nas import NASRepos
from ..prefix import DataPath
from ..prefix import GitReposPath
from ..prefix import NASReposPath
from ..prefix import TargetPath
from .adaptor import CSVAdapter
from .adaptor import KerasAdapter
from .adaptor import KerasAdapterWeights
from .adaptor import KmeansAdapter
from .adaptor import ScalerAdapter



class ReposFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master):
        repos = {}
        for item in master.repository(lookup='all'):
            
            # Git repos
            if item.type == 'git':
                gpath =  GitReposPath(master, item)
                grepos = GitRepos(master, gpath.uri(),  gpath.branch(), gpath.local())
                repos[item.name] = grepos
                
            if item.type == 'nas':
                gpath = NASReposPath(master, item)
                grepos = NASRepos(master, local=gpath.local())
                repos[item.name] = grepos
            
        # # NAS repos default.
        # if not 'nas' in list(repos.keys()):
        #     gpath = NASReposPath(master)
        #     grepos = NASRepos(master, local=gpath.local())
        #     repos['nas'] = grepos
                
        return repos



class AttachFiles(object):    
    def __init__(self, master, entry, prefix):
        self.master = master
        self.entry = entry
        self.files = {}
        self.remotes = {}
        self.prefix = prefix
            
    def apply(self, repos):
        # append file to use repository
        for meta in self.entry:
            _name_, _path_ = prefix.path.get_target_path(meta)            
            
            # set unique values
            if not _name_ in self.files.keys():
                self.files[_name_] = []
            
            remote = repos[_name_]
            filepath = []
            for path in _path_:
                for file in self.prefix(self.master, path):
                    filepath.append(file)
            
            filepath = list(set(filepath))
            for file in filepath:
                if not remote.exists(file):
                    remote.add(file)
                new_meta = meta.copy()
                new_meta.uri = file
                self.files[_name_].append(new_meta)

            self.remotes[_name_] = remote
            # self.files[_name_] = list(set(self.files[_name_]))


class AdaptorBuilder(object):
    
    def __init__(self, master):
        self.master = master
        self.builders = []
        self.adaptors = {'keras':[], 'kmeans': [], 'scaler': [], 'database': []}
        
    def __call__(self, file, remote, option):
        self.file = file 
        self.remote = remote
        self.option = option
        return self
    
    def build(self):
        raise NotImplementedError
    
    def append(self, builder):
        self.builders.append(builder)
                
    def update(self, entry):
        for name, item in entry.items():
            self.adaptors[name].extend(item)
            
    def __getitem__(self, index):
        if index == 0:
            return self.adaptors
        elif index == 1:
            return self.remote


class ImportAdaptorBuilder(AdaptorBuilder):
    
    def __init__(self, master):
        super().__init__(master)
        self.entry = master.import_model(lookup='all')
    
    def build(self, repos):
        self.remote = repos
        if self.entry:
            attacher = AttachFiles(self.master, self.entry, TargetPath())
            attacher.apply(repos)        
            for actor in self.builders:
                actor = actor(attacher.files, attacher.remotes, "import")
                self.update(actor.build())
    
class ExportAdaptorBuilder(AdaptorBuilder):
    
    def __init__(self, master):
        super().__init__(master)
        self.entry = master.export_model(lookup='all')
    
    def build(self, repos):
        self.remote = repos
        if self.entry:
            attacher = AttachFiles(self.master, self.entry, TargetPath())
            attacher.apply(repos)        
            for actor in self.builders:
                actor = actor(attacher.files, attacher.remotes, "export")
                self.update(actor.build())


class DatabaseAdaptorBuilder(AdaptorBuilder):
    
    def __init__(self, master):
        super().__init__(master)
        self.entry = master.database(lookup='all')
    
    def build(self, repos):
        self.remote = repos
        if self.entry:
            attacher = AttachFiles(self.master, self.entry, DataPath())
            attacher.apply(repos)        
            for actor in self.builders:
                actor = actor(attacher.files, attacher.remotes, "import")
                self.update(actor.build())

    
class KerasAdaptorBuilder(AdaptorBuilder):
    
    def __init__(self, master):
        super().__init__(master)
        self.adapter = {'keras':[]}
    
    def build(self):
        for key, repos in self.remote.items():        
            for value in self.file[key]:
                if not hasattr(value, 'option'):
                    continue
                
                if not hasattr(value, 'algorithm'):
                    continue
                
                if value.algorithm in ['word2vec', 'BiLSTM'] and value.option == 'legacy':
                    self.adapter['keras'].append(
                        KerasAdapter(
                            self.master,
                            # prefix.model.name(value.uri),
                            Path(value.uri),
                            repos,
                            self.option
                        )
                    )
        return self.adapter
        
class KerasAdapterWeightsBuilder(AdaptorBuilder):
    
    def __init__(self, master):
        super().__init__(master)
        self.adapter = {'keras':[]}
    
    def build(self):
        for key, repos in self.remote.items():        
            for value in self.file[key]:
                if hasattr(value, 'option'):
                    continue
                
                if not hasattr(value, 'algorithm'):
                    continue
                
                if value.algorithm in ['BiLSTM']:
                    self.adapter['keras'].append(
                        KerasAdapterWeights(
                            self.master,
                            # prefix.model.name(value.uri),
                            Path(value.uri),
                            repos,
                            self.option
                        )
                    )
        return self.adapter
    

class KmeansAdapterBuilder(AdaptorBuilder):
    
    def __init__(self, master):
        super().__init__(master)
        self.adapter = {'kmeans':[]}
    
    def build(self):
        for key, repos in self.remote.items():        
            for value in self.file[key]:
                if not hasattr(value, 'algorithm'):
                    continue
                
                if value.algorithm in ['kmeans']:
                    self.adapter['kmeans'].append(
                        KmeansAdapter(
                            self.master,
                            # prefix.model.name(value.uri),
                            Path(value.uri),
                            repos,
                            self.option
                        )
                    )
        return self.adapter


class ScalerAdapterBuilder(AdaptorBuilder):
    
    def __init__(self, master):
        super().__init__(master)
        self.adapter = {'scaler':[]}
    
    def build(self):
        for key, repos in self.remote.items():        
            for value in self.file[key]:
                if not hasattr(value, 'algorithm'):
                    continue
                
                if value.algorithm in ['z-score', 'StandardScaler']:
                    self.adapter['scaler'].append(
                        ScalerAdapter(
                            self.master,
                            # prefix.model.name(value.uri),
                            Path(value.uri),
                            repos,
                            self.option
                        )
                    )
        return self.adapter


class CSVAdapterBuilder(AdaptorBuilder):
    
    def __init__(self, master):
        super().__init__(master)
        self.adapter = {'database':[]}
    
    def build(self):
        for key, repos in self.remote.items():        
            for value in self.file[key]:
                if not hasattr(value, 'type'):
                    continue
                
                if value.type in ['csv']:
                    self.adapter['database'].append(
                        CSVAdapter(
                            self.master,
                            # prefix.model.name(value.uri),
                            Path(value.uri),
                            repos,
                            self.option
                        )
                    )
        return self.adapter

    
    