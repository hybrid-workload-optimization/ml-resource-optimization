import json
import requests

from flask import Flask, jsonify, request, Response
#from flask_restful import Api
#from flask_restful import Resource
from flask_restful import reqparse
from flask_restx  import Api, Resource

from ..utils import logger, MetaclassSingleton, serialize

def response_ok(message=None):
    return "ok." if message is None else message, 200

def request_post(url, data):
    error = None
    #headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    headers = {'Connection': 'keep-alive', 'Content-type': 'application/json', 'Accept': '*/*'}
    try:
        return requests.post(
            data=json.dumps(
                data,
                default=serialize.json_serialize,
                indent=4),
            url=url,
            headers=headers)

    except requests.exceptions.HTTPError as errh:
        error = f"Http Error: {errh}"
        
    except requests.exceptions.ConnectionError as errc:
        error = f"Error Connecting: {errc}"
        
    except requests.exceptions.Timeout as errt:
        error = f"Timeout Error: {errt}"
        
    except requests.exceptions.RequestException as err:
        error = f"OOps: Something Else {err}"
        
    raise Exception(error)


class APIResource(metaclass=MetaclassSingleton):

    def __init__(self):
        self.log = logger.get("APIResource")
        
    def setResource(self, resource):
        self.resource = resource

    def count(self):
        return len(self.resource)

    def get(self, index):
        return self.resource[index]

    def getAPI(self, endpoint):
        for item in self.resource:
            for k in item.keys():
                if k == 'api':
                    for v in item['api']:
                        if v == endpoint:
                            return item
        return None

    def check_endpoint(self, endpoint):
        # item = self.getAPI(endpoint)
        # return False if item is None else True

        entry = self.resource['api']
        if not isinstance(entry, list):
            entry = [entry]

        for api in entry:
            if api == endpoint:
                return True
        return False

    def getMethod(self, method):
        for item in self.resource:
            if item['method'] == method:
                return item
        return None
        
    def getTemplate(self):
        for item in self.resource:
            for k in item.keys():
                if k == 'template':
                    return item['template']
        return None
    
    def isModel(self, item, model):
        try:
            for a in item['model']:
                if a == model:
                    return True
                    
        except Exception as e:
            return False

        return False





class RESTful(metaclass=MetaclassSingleton):
    
    def __init__(self):
        self.log = logger.get("RESTful")
        self.app = Flask('RESTful')
        self.api = Api(self.app)
        
    def add_resource(self, object, api):
        self.api.add_resource(object, api)

    def start(self, port=5050):
        self.log.info(f"Serving RESTful API. (Running on http://0.0.0.0:{port}/)")
        self.app.run(host='0.0.0.0', port=port, debug=False)
        
    def join(self):
        pass

    def build(self, func, *args, **kargs):
        self.__func__ = (func, args, kargs)
        func(*args, **kargs)

