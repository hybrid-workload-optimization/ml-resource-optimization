
from ..utils import logger
from .dataset import CMPDataSet
from .trainset import TrainSet
from .testset import TestSet

class DatasetFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master, searcher, preprocessor=None, action="dataset"):
        
        if action == 'dataset':
            db = master.database()
            if db.driver == "cmplog":
                return CMPDataSet(searcher, preprocessor)

        if action == 'sampling':
            return TrainSet(master, searcher)
        
        if action == 'forecast':
            return TestSet(master, searcher)
        
        raise Exception(f"Unknown the value of action variable. (action={action})")