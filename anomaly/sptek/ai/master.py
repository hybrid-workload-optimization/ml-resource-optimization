import os
import yaml
from munch import Munch

from .utils import MetaclassSingleton, logger
 
def updateMuch(config):
    munch = Munch()
    for key, value in config.items():
        if isinstance(value, dict):
            munch[key] = updateMuch(value)
        else:
            munch[key] = value
    return munch

class MasterParser():
    
    def __init__(self):
        self.log = logger.get(self)

    def parse(self, fp):
        self.data = yaml.load(fp, Loader=yaml.FullLoader)
    
    def version(self):
        try:
            return self.data['version']
        except:
            return None

    def service(self, key):
        try:
            return self.data['services'][key]
        except:
            return None

    def restful(self, key):
        try:
            return self.service('restful')
        except:
            return None

    def export(self):
        try:
            return self.service('exports')
        except:
            return None

    def database(self):
        try:
            return self.service('database')
        except:
            return None

    def periods(self):
        try:
            return self.service('periods')
        except:
            return None

    def filters(self):
        try:
            return self.service('filters')
        except:
            return None

    def features(self):
        try:
            return self.service('features')
        except:
            return None

    def periods(self):
        try:
            return self.service('periods')
        except:
            return None

    def generators(self):
        try:
            return self.service('generators')
        except:
            return None

    def restfuls(self):
        try:
            return self.service('restful')
        except:
            return None

    def preprocessing(self):
        try:
            return self.service('preprocessing')
        except:
            return None

    def repository(self):
        try:
            return self.service('repository')
        except:
            return None

    def hyperparameter(self):
        try:
            return self.service('hyperparameter')
        except:
            return None

    def earlystopping(self):
        try:
            return self.service('earlystopping')
        except:
            return None

    def ip(self):
        try:
            return self.service('ip')
        except:
            return None

    def port(self):
        try:
            return self.service('port')
        except:
            return None

    def gpu(self):
        try:
            return self.service('gpu')
        except:
            return None

    def worker(self):
        try:
            return self.service('worker')
        except:
            return None

    def route(self):
        try:
            return self.service('route')
        except:
            return None

    def sampling(self):
        try:
            return self.service('sampling')
        except:
            return None

    def routes(self):
        try:
            return self.service('routes')
        except:
            return None

    def model(self):
        try:
            return self.service('model')
        except:
            return None


class Master(metaclass=MetaclassSingleton):
     
    def __init__(self, path):
        self.log = logger.get(self)
        self.parser = MasterParser()
        self.path = path
       
    def hasConfig(self):
        self.error_ = False if os.path.exists(self.path) else True
        return self.error_ == False

    def load(self):
        if self.hasConfig() is False:
            self.log.info(f"not exsits master file.")
            return

        with open(self.path) as fp:
            self.parser.parse(fp)

        self.log.info(f"load master file.")

    def port(self):
        return self.parser.port()

    def database(self):
        return updateMuch(self.parser.database())

    def model(self):
        return updateMuch(self.parser.model())

