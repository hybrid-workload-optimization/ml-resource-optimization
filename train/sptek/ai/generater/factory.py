from .timestamp import Timestamp
from .weekday import WeekdayGenerator
from ..utils import logger

class GeneratorFactory(object):
    
    @classmethod
    def create(cls, master):
        meta = master.machine(dtype='generators')
        # logger.debug(f"{meta}")
        
        if meta.generator == 'WeekdayGenerator':
            return WeekdayGenerator(master)


class TimestampFactory(object):
    
    @classmethod
    def create(cls, master, time):
        meta = master.machine(dtype='features')
        key = meta.key
        
        meta = master.machine(dtype='forecast')
        meta['key'] = key
        
        return Timestamp(master, time, meta)