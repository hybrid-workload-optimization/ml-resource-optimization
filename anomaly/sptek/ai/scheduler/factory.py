from datetime import datetime
from ..utils import logger
from .scheduler import Scheduler, JobTriggerParameterBuilder

class SchedulerFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master, name):
        meta = master.scheduler(name=name)
        log = logger.get('Scheduler')
        log.info(meta)
        
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