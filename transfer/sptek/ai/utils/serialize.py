import numpy as np
def json_serialize(obj):
    """JSON serializer for objects not serializable by default json code"""

    # if isinstance(obj, Adaptor):
    #     return str(obj)
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, object):
        return str(obj)

    return obj.__dict__

def to_vector(s):
    s = s.strip('')
    s = s.strip('[')
    s = s.strip(']')
    s = s.split(', ')
    l = list(map(float, s))
    return l
    

class reprint():
    def __init__(self, msg):
        self.msg = msg
        
    def __repr__(self):
        return self.msg
    
