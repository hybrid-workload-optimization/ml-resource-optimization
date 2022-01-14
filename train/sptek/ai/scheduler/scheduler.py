from pytz import utc
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .. import master


def _update_params_(params, meta):
    for key in list(meta.keys()):
        if key in params:
            params[key] = meta[key]
    return params
    
class JobTriggerParameterBuilder:

    @staticmethod
    def build_cron_type_params(meta):
        params = {}
        params.setdefault("trigger", "cron")
        params.setdefault("year", "*")
        params.setdefault("month", "*")
        params.setdefault("day", "*")
        params.setdefault("hour", "*")
        params.setdefault("minute", "*")
        params.setdefault("second", "*")
        params.setdefault("week", "*")
        params.setdefault("day_of_week", "*")
        params.setdefault("start_date", "*")
        params.setdefault("end_date", "*")
        params.setdefault("enable", "true")
        return master.createMuch(_update_params_(params, meta))
    
    @staticmethod
    def build_interval_type_params(meta):
        params = {}
        params.setdefault("trigger", "interval")
        params.setdefault("days", 0)
        params.setdefault("hours", 0)
        params.setdefault("minutes", 0)
        params.setdefault("seconds", 0)
        params.setdefault("weeks", 0)
        params.setdefault("start_date", None)
        params.setdefault("end_date", None)
        params.setdefault("enable", "true")
        return master.createMuch(_update_params_(params, meta))
    
    @staticmethod
    def build_date_type_params(meta):
        params = {}
        params.setdefault("trigger", "date")
        params.setdefault("run_date", datetime.now())        
        params.setdefault("enable", "true")
        return master.createMuch(_update_params_(params, meta))



class JobTrigger(object):
    
    @classmethod
    def job():
        raise NotImplementedError
    

class Scheduler(object):
    
    def __init__(self, option=None):
        self.option=option
        self.scheduler = BackgroundScheduler()
        self.scheduler.configure(timezone=utc)
        
    def enable(self):
        if not hasattr(self.option, 'enable'):
            return False        
        return self.option.enable
            
    def start(self):
        if self.enable():
            self.scheduler.start()
        
    def _add_job_cron_(self, job, option):
        if hasattr(option, 'from_crontab'):
            self.scheduler.add_job(job, CronTrigger.from_crontab(option.from_crontab, timezone=utc))
            
        else:
            self.scheduler.add_job(job, option.trigger, year=option.year, month=option.month,
                                   day=option.day, week=option.week, day_of_week=option.day_of_week,
                                   hour=option.hour, minute=option.minute, second=option.second)
        
    def _add_job_interval_(self, job, option):
        self.scheduler.add_job(job, option.trigger, weeks=option.weeks, days=option.days,
                               hours=option.hours, minutes=option.minutes, seconds=option.seconds)
        
    def _add_job_date_(self, job, option):
        self.scheduler.add_job(job, option.trigger, run_date=option.run_date)

    def add_job(self, func):
        
        if self.option.trigger == 'cron':
            return self._add_job_cron_(func.job, self.option)
        
        if self.option.trigger == 'interval':
            return self._add_job_interval_(func.job, self.option)
        
        if self.option.trigger == 'date':
            return self._add_job_date_(func.job, self.option)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
