from datetime import datetime

from ..searcher import SearcherFactory
from ..scheduler import JobTrigger
from ..utils import MetaclassSingleton, logger
from ..dataset import DatasetFactory, DataSearcher
from ..prefix import ClusterPath
from ..ml import ClusterFactory
from ..bucket import BucketFactory

class DataServing(metaclass=MetaclassSingleton):
    
    def __init__(self):
        self.log = logger.get(self)
        self.serving = {}
    
    def attach(self, name, serve):
        """This function is attach for use serving adaptors.

        Args:
            serve (Serving): This 'serve' parameter is requst the input a Serving subclass object.
        """
        
        if not name in self.serving.keys():
            self.serving[name] = []
        
        if not isinstance(serve, list):
            serve = [serve]
            
        for item in serve:
            self.serving[name].append(item)
    
    def get(self, action, log_type=None):
        """This get function is returns a data adapter based on the action.

        Args:
            action (str): this action parameter is input "train" or "forecast" value.
        """
        if log_type is None:
            return self.serving[action]
        
        for adaptor in self.get(action):
            if adaptor.log_type == log_type:
                return adaptor
    
    def entry(self):
        return list(self.serving.keys())
    
    def start(self):
        entry = self.entry()
        entry.remove('scheduler')
        scheduler = self.get('scheduler')[0]        
        for name in entry:
            for adaptor in self.get(name):
                scheduler.add_job(adaptor)        
        scheduler.start()

class Serving(JobTrigger):
    
    @property
    def action(self):
        raise NotImplementedError
    
    @property
    def log_type(self):
        raise NotImplementedError
    
    def get(self, log_type):
        raise NotImplementedError
    
    def job(self):
        raise NotImplementedError
    
class Adaptor(Serving):
    
    def __init__(self, master, log_type, action):
        self.log = logger.get(self)
        self.master = master
        self._action_ = action
        self._log_type_ = log_type
    
    @property
    def action(self):
        return self._action_
    
    @property
    def log_type(self):
        return self._log_type_
    
    
class TrainAdaptor(Adaptor):
    
    def __init__(self, master, log_type):
        super().__init__(master, log_type, "train")
        self.log = logger.get(self)
        self.searcher = None
        self.dataset = None
        self.dateTime = datetime.now()
        
    def connect(self):
        
        # create data searcher
        self.searcher = SearcherFactory.create(self.master, model=self.log_type, dt=self.dateTime, dtype='train')
        self.searcher.fork()
        
        # # build train dataset
        self.dataset = DatasetFactory.create(self.master, self.searcher, None)
        self.dataset.fork_build(fork=1)                
        self.dataset.wait_completed()

    def job(self, dateTime=None):
        # run searcher job
        if self.searcher:
            self.searcher.job(dateTime=dateTime)
        
        if self.dataset:
            self.dataset.job()
        
        
        
        

class ForecastAdaptor(Adaptor):
    
    def __init__(self, master, log_type):
        super().__init__(master, log_type, "forecast")
        self.log = logger.get(self)
        self.searcher = None
        self.dataset = None
        self.cluster = None
        self.bucket = None
        self.dateTime = datetime.now()
        
    def connect(self):        
        # create data searcher
        self.searcher = SearcherFactory.create(self.master, model=self.log_type, dt=self.dateTime, dtype='forecast')
        self.searcher.fork()
    
        # build forecast dataset
        self.dataset = DatasetFactory.create(self.master, self.searcher, None)
        self.dataset.fork_build(fork=1)
        self.dataset.wait_completed()
        
        # load clustering model
        cluster_path  = ClusterPath(self.master)
        self.cluster = ClusterFactory.create(self.master, DataSearcher(self.dataset), action="forecast")
        self.cluster.load(cluster_path.path())
        
        # create term record
        self.bucket = BucketFactory.create(self.master, cluster=self.cluster, searcher=DataSearcher(self.dataset))
        self.bucket.fork_fit(1)
        self.bucket.wait_completed()

    def job(self, dateTime=None):
        # run searcher job
        if self.searcher:
            self.searcher.job(dateTime=dateTime)
        
        if self.dataset:
            self.dataset.job()
            
        if self.bucket:
            self.bucket.job()
