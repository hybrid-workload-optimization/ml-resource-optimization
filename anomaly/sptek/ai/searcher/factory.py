
from ..utils import logger
from .cmp import CMPSearcher

class SearcherFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master, dtype, parser=None, model=None, dt=None, interval=None):
        log = logger.get(cls.__name__)
        
        if model is None and parser is None:
            raise Exception('SearcherFactory -- request input model or parser parameter') 
        
        if model is None:
            model = parser.model()
        
        db = master.database(dtype=model)
        # logger.debug(db)

        # create data searcher filter.
        if dt is None and parser is None:
            raise Exception('SearcherFactory -- request input dt(datetime) or parser parameter') 
        
        if dt is None:
            dt = parser.dateTime()
        meta = master.searcher(dtype=dtype, lookup="match")
        meta['current'] = dt
        
        t = master.scheduler(name='searcher', lookup="match")
        if interval is None:
            interval = SearcherFactory.convert_time(t)
        meta['interval'] = interval
        
        if db.driver == "cmplog":
            searcher = CMPSearcher(master, db, meta)
            searcher(meta)
            return searcher
    
    @classmethod
    def convert_time(cls, t):
                
        if hasattr(t, 'week'):
            return str(t.week) + 'w'
        
        if hasattr(t, 'day'):
            return str(t.day) + 'd'
        
        if hasattr(t, 'hour'):
            return str(t.hour) + 'h'
        
        if hasattr(t, 'minute'):
            return str(t.minute) + 'm'
        
        if hasattr(t, 'second'):
            return str(t.second) + 's'
      
        if hasattr(t, 'weeks'):
            return str(t.weeks) + 'w'
        
        if hasattr(t, 'days'):
            return str(t.days) + 'd'
        
        if hasattr(t, 'hours'):
            return str(t.hours) + 'h'
        
        if hasattr(t, 'minutes'):
            return str(t.minutes) + 'm'
        
        if hasattr(t, 'seconds'):
            return str(t.seconds) + 's'