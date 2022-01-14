from ..utils import logger

class RequestParser():
    
    def __init__(self, raw):
        self.log = logger.get("RequestParser")
        self.raw = raw

    def get(self, key):
        try:
            return self.raw[key]
        except Exception as e:
            self.log.error(e)
        return None

    def model(self):
        return self.get("model")

    def period(self):
        return self.get("period")

    def dateTime(self):
        return self.get("dateTime")
    
    def __setitem__(self, key, value):
        self.raw[key] = value