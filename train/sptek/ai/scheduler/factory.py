from datetime import datetime
from .scheduler import JobTriggerParameterBuilder
from .scheduler import Scheduler
from ..utils import logger


class SchedulerFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master, name):
        meta = master.scheduler(name=name)
        # logger.debug(meta)
        
        if meta is None:
            return None
        
        if meta.type == 'cron':            
            trigger = JobTriggerParameterBuilder.build_cron_type_params(meta)
            if hasattr(meta, 'from_crontab'):
                trigger.from_crontab = meta.from_crontab
        
        elif meta.type == 'interval':
            trigger = JobTriggerParameterBuilder.build_interval_type_params(meta)
        
        elif meta.type == 'date':
            trigger = JobTriggerParameterBuilder.build_date_type_params(meta)

        return Scheduler(trigger)