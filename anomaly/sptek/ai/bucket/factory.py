
from ..utils import logger
from .aggregation import TermAggregation
from .searcher import TermSearcher

class BucketFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master, cluster, searcher):
        meta = master.machine(dtype="forecast")
        
        if meta.type == "forecast":
            return TermAggregation(master, cluster, BucketFactory.searcher(master, embed=searcher))

    @classmethod
    def searcher(cls, master, embed):
        meta = master.machine(dtype="forecast")
        
        if meta.type == "forecast":
            return TermSearcher(master, embed, meta.term)
        
