from . import rest
from ..utils import logger


class Requester(object):
    
    def __init__(self, uri):
        self.log = logger.get(self)
        self.uri = uri

    def request(self, body):
        error = None
        try:
            # logger.debug(body)
            response = rest.request_post(self.uri, body)
            response.raise_for_status()
            body = response.json()
            if not body:
                body = response.text
            return body 

        except TypeError as err:
            error = f"Error: Request Type: {err}"
            self.log.error(error)

        except Exception as err:
            error = err
            self.log.error(error)
        
        raise Exception(error)