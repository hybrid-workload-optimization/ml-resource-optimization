
from ..utils import logger
from .anomalize import StreamDetector

class AnomalyFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master):
        return StreamDetector(master)

