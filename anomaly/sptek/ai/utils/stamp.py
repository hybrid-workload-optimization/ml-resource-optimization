import re
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
    
    @classmethod
    def build(self, start, end, tick='hours', step=1, format="%Y-%m-%d %H:%M:%S"):
        _start_ = start
        _end_ = end
        if isinstance(start, str):
            _start_ = datetime.strptime(start, format)
        if isinstance(end, str):
            _end_ = datetime.strptime(end, format)
        
        stemp = []
        delta = getTimedelta(tick_map[tick], int(step))
        _current_ = _start_
        while _current_ <= _end_:
            _current_ = _current_ + delta
            stemp.append(_current_)
        return stemp

    @classmethod
    def strptime(slef, t, format="%Y-%m-%d %H:%M:%S"):
        return datetime.strptime(t, format) 

    @classmethod
    def next(self, t, tick='hours', step=1):
        if isinstance(t, str):
            return TimeGenerator.strptime(t) + getTimedelta(tick, step)
        return t + getTimedelta(tick, step)

    @classmethod
    def prev(self, t, tick='hours', step=1):
        if isinstance(t, str):
            return TimeGenerator.strptime(t) - getTimedelta(tick, step)
        return t - getTimedelta(tick, step)
        
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

tick_map = {
    'us': 'microseconds',
    'ms': 'milliseconds',
    's': 'seconds',
    'm': 'minutes',
    'h': 'hours',
    'd': "days",
    'w': "weeks"
}

tick_priority = {
    'us': 0,
    'ms': 1,
    's' : 2,
    'm' : 3,
    'h' : 4,
    'd' : 5,
    'w' : 6
}

def _tick_and_nubmer_(v):
    t = re.sub(r'[^a-z]', '', v)
    n = re.sub(r'[^0-9]', '', v)
    return (t, n)

def trim_time(t, f):
    tick, _ = _tick_and_nubmer_(f)    
    tick = tick_map[tick]
    
    if tick == 'seconds' :
        t = t.replace(microsecond=0)
    
    if tick == 'minutes' :
        t = t.replace(second=0, microsecond=0)
    
    if tick == 'hours' :
        t = t.replace(minute=0, second=0, microsecond=0)
    
    if tick == "days" :
        t = t.replace(hour=0, minute=0, second=0, microsecond=0)
    
    return t
    
    # replace(
    #     year=self.year, 
    #     month=self.month, 
    #     day=self.day, 
    #     hour=self.hour, 
    #     minute=self.minute, 
    #     second=self.second, 
    #     microsecond=self.microsecond, 
    
    

def prev_time(t, f):
    # tick = re.sub(r'[^a-z]', '', f)
    # numbers = re.sub(r'[^0-9]', '', f)
    # return TimeGenerator.prev(t, tick_map[tick], int(numbers))
    tick, numbers = _tick_and_nubmer_(f)
    return TimeGenerator.prev(t, tick_map[tick], int(numbers))
    

def next_time(t, f):
    # tick = re.sub(r'[^a-z]', '', f)
    # numbers = re.sub(r'[^0-9]', '', f)
    # return TimeGenerator.next(t, tick_map[tick], int(numbers))
    tick, numbers = _tick_and_nubmer_(f)
    return TimeGenerator.next(t, tick_map[tick], int(numbers))

def strptime(t):
    return TimeGenerator.strptime(t)

def scroll_delta(val, to):
    _va_t_, _va_n_ = _tick_and_nubmer_(val)
    _to_t_, _to_n_ = _tick_and_nubmer_(to)
    
    _va_d_ = getTimedelta(tick_map[_va_t_], int(_va_n_))
    _to_d_ = getTimedelta(tick_map[_to_t_], int(_to_n_))
    delta = _va_d_ / _to_d_
    
    return delta, _to_t_

def get_delta(val):
    _t_, _n_ = _tick_and_nubmer_(val)
    _d_ = getTimedelta(tick_map[_t_], int(_n_))
    return _d_, _t_

def scroll_delta_dev(val, to):
    _va_d_, _ = get_delta(val)
    _to_d_, _to_t_ = get_delta(to)
    delta = _va_d_ / _to_d_
    return delta, _to_t_

def scroll_delta_mul(val, to):
    _va_d_, _ = get_delta(val)
    _to_d_, _to_t_ = get_delta(to)
    delta = _va_d_ * _to_d_
    return delta, _to_t_