import re
import os
from numpy.lib.arraysetops import isin
import yaml

from numpy.lib import histograms
from munch import Munch
from munch import munchify
from pathlib import Path

from .utils import logger
from .utils import MetaclassSingleton
from .utils import path as utl_path


def createMuch(config):
    return munchify(config)

def updateMuch(config):
    return munchify(config)

def lookupMuch(munch, key, value, lookup='first'):
    if not munch:
        return None
    
    for item in munch:
        if key == 'name' and item.name == value:
            return item

        if key == 'type':
            if isinstance(item.type, list):
                if value in item.type:
                    item.type = value
                    return item
            if value == item.type:
                return item
        
    if len(munch) and lookup == 'first':
        return munch[0]


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
        
    def service(self, tocken):
        return self.parser.service(tocken)

    def ip(self):
        return self.parser.ip()

    def port(self):
        return self.parser.port()
    
    def service_type(self):
        return self.parser.service_type()
    
    # lookup : first | match | all
    def _search_(self, data, name=None, dtype=None, lookup='first'):
        munch = updateMuch(data)
        if dtype is not None:
            return lookupMuch(munch, 'type', dtype, lookup)
        if lookup in ['first', 'match']:
            return lookupMuch(munch, 'name', name, lookup)
        return munch
    
    def database(self, name=None, dtype=None, lookup='first'):
        return self._search_(self.parser.database(), name, dtype, lookup)

    def model(self):
        return updateMuch(self.parser.model())

    def period(self, tp=None):
        return lookupMuch(updateMuch(self.parser.period()), 'type', tp)

    def endpoint(self):
        return self.parser.endpoint()

    def route(self):
        return self.parser.route()

    def searcher(self, name=None, dtype=None, lookup='first'):
        return self._search_(self.parser.searcher(), name, dtype, lookup)

    def preprocess(self, name=None, dtype=None):
        if dtype is not None:
            return lookupMuch(updateMuch(self.parser.preprocess()), 'type', dtype)
        return lookupMuch(updateMuch(self.parser.preprocess()), 'name', name)

    def import_model(self, name=None, dtype=None, lookup='first'):
        return self._search_(self.parser.import_model(), name, dtype, lookup)
    
    def export_model(self, name=None, dtype=None, lookup='first'):
        return self._search_(self.parser.export_model(), name, dtype, lookup)
    
    def machine(self, name=None, dtype=None, lookup='first'):
        return self._search_(self.parser.machine(), name, dtype, lookup)
    
    def repository(self, name=None, dtype=None, lookup='first'):
        return self._search_(self.parser.repository(), name, dtype, lookup)
    
    def scheduler(self, name=None, dtype=None, lookup='first'):
        return self._search_(self.parser.scheduler(), name, dtype, lookup)

    def realpath(self, uri):
        if hasattr(self, 'temp_dir'):
            export = Path(utl_path.uri_to_absolutized(uri))
            export = str(export).replace('/{temp}', '.')
            return Path(self.temp_dir).joinpath(export)

    def target(self, name=None, dtype=None, lookup='first'):
        return self._search_(self.parser.target(), name, dtype, lookup)


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

    def period(self):
        try:
            return self.service('period')
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

    def preprocess(self):
        try:
            return self.service('preprocess')
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

    def endpoint(self):
        try:
            return self.service('endpoint')
        except:
            return None

    def searcher(self):
        try:
            return self.service('searcher')
        except:
            return None

    def import_model(self):
        try:
            return self.service('import')
        except:
            return None
    
    def export_model(self):
        try:
            return self.service('export')
        except:
            return None
        
    def machine(self):
        try:
            return self.service('machine')
        except:
            return None
        
    def repository(self):
        try:
            return self.service('repository')
        except:
            return None
        
    def scheduler(self):
        try:
            return self.service('scheduler')
        except:
            return None
        
    def service_type(self):
        try:
            return self.service('type')
        except:
            return None
        
    def target(self):
        try:
            return self.service('target')
        except:
            return None
