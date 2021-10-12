
from ..utils import logger
from .cmp import CMPSearcher

class SearcherFactory(object):

    def __init__(self):
        self.log = logger.get(self)

    @classmethod
    def create(cls, master):
        db = master.database()
        logger.debug(db)

        if db.driver == "cmplog":
            return CMPSearcher(db)