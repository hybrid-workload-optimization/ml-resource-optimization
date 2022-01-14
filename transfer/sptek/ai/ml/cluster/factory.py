from .cluster import KMeansCluster
from ...utils import logger

class ClusterFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master, searcher, action="train"):
        
        meta = master.import_model(dtype='cluster')
        n_clusters = 'auto'
        
        if action == 'train':
            meta = master.machine(dtype='cluster')
            n_clusters = meta.clusters
        
        if meta.algorithm == "kmeans":
            return KMeansCluster(master,
                                searcher=searcher,
                                clusters=n_clusters,
                                export_path=None, # export_path=export_path,
                                import_path=None) # import_path=import_path)
