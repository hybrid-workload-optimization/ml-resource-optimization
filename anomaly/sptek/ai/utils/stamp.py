import time
from datetime import datetime, timedelta

from . import logger

def nowstamp():
    now = datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d')
    return nowDatetime

def msec(t):
    return int(round(t * 1000))

# days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0

def getTimedelta(stemp, tick):
    if stemp == "microseconds":
        return timedelta(microseconds=tick)

    if stemp == "milliseconds":
        return timedelta(milliseconds=tick)

    if stemp == "seconds":
        return timedelta(seconds=tick)
    
    if stemp == "minutes":
        return timedelta(minutes=tick)

    if stemp == "hours":
        return timedelta(hours=tick)

    if stemp == "days":
        return timedelta(days=tick)

    if stemp == "weeks":
        return timedelta(weeks=tick)

    return None


class TimeGenerator():

    def __init__(self, x, start, prev=False, tick='hours', step=1, format="%Y-%m-%d %H:%M:%S"):
        self.stemp = []
        now = datetime.strptime(start, format)        
        if prev is True:
            for y in range(x):
                self.stemp.append(now.strftime(format))
                now = now - getTimedelta(tick, step)
            self.stemp = list(reversed(self.stemp))
        else:
            for y in range(x):
                now = now + getTimedelta(tick, step)
                self.stemp.append(now.strftime(format))

    def get(self):
        return self.stemp

def timestamp_to_datetime(ts):
    if isinstance(ts, str):
        ts = float(ts)
    return datetime.fromtimestamp(
        ts
    )

def datetime_to_timestamp(dt):
    return time.mktime(dt.timetuple())
    
def current_time():
    return datetime.now()