import re

def type(uri):
    try:
        uri = str(uri)
        sp = re.split('://', uri)
        if len(sp) > 1:
            return sp[0]
        return 'nas'
    except:
        return 'nas'
    
def name(uri):
    try:
        uri = str(uri)
        if re.search('://', uri):
            sp = re.split('/', uri)
            return sp[2]
        return 'nas'
    except:
        return 'nas'