
def json_serialize(obj):
    """JSON serializer for objects not serializable by default json code"""

    # if isinstance(obj, Adaptor):
    #     return str(obj)

    if isinstance(obj, object):
        return str(obj)

    return obj.__dict__


class reprint():
    def __init__(self, msg):
        self.msg = msg
        
    def __repr__(self):
        return self.msg