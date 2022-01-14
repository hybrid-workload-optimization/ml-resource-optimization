
from .master import Master

def model_entry(master):
    db = master.database()
    if db.driver == 'cmplog':
        return db.type

def model_period(master, tp=None):
    return master.period(tp="output").month

def model_window(master, tp=None):
    pass