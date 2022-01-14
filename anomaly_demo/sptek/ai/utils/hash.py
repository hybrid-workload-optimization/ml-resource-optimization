import hashlib
import uuid

def RandomHash():
    id = uuid.uuid4()
    hm = hashlib.md5()
    hm.update(str(id).encode('utf-8'))
    return hm.hexdigest()
