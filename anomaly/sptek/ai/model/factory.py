
from ..utils import logger
from .word2vec import EmbeddingVector

class ModelFactory(object):

    def __init__(self):
        self.log = logger.get(self)

    @classmethod
    def create(cls, master):
        meta = master.model()
        logger.debug(meta)

        if meta.architecture == "word2vec":
            return EmbeddingVector(meta)