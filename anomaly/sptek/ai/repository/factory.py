
from .. import prefix
from .git import GitRepos
from .nas import NASRepos
from ..prefix import GitReposPath
from ..prefix import NASReposPath
from .adaptor import KerasAdapter
from .adaptor import KerasAdapterWeights
from .adaptor import KmeansAdapter
from .adaptor import ScalerAdapter


class ReposFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master):
        repos = cls.build_repos(master)
        adaptor = cls.build_adaptor(master, repos)
        return (adaptor, repos)
    
    @classmethod
    def build_repos(cls, master):
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
            
    @classmethod
    def build_adaptor(cls, master, repos):
        
        adapter = {'keras':[], 'kmeans': [], 'scaler': []}
        for meta in master.import_model(lookup='all'):
            _r_name_ = prefix.repos.name(meta.uri)
            _m_name_ = prefix.model.name(meta.uri)
            
            # append file to use repository
            repos[_r_name_].add(meta.uri)
            
            # create adaptor
            if meta.algorithm in ['word2vec']:
                adapter['keras'].append(KerasAdapter(master, _m_name_, repos[_r_name_]))
                
            if meta.algorithm in ['BiLSTM']:
                adapter['keras'].append(KerasAdapter(master, _m_name_, repos[_r_name_]))
                # adapter['keras'].append(KerasAdapterWeights(master, _m_name_, repos[_r_name_]))
            
            if meta.algorithm in ['kmeans']:
                adapter['kmeans'].append(KmeansAdapter(master, _m_name_, repos[_r_name_]))
                
            if meta.algorithm in ['z-score']:
                adapter['scaler'].append(ScalerAdapter(master, _m_name_, repos[_r_name_]))

        return adapter
