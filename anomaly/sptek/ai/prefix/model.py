import re
from pathlib import Path
from ..utils import path as path_util

def name(uri):
    try:
        return Path(path_util.uri_to_normpath(uri)).relative_to('/')         
    except Exception as e:
        return None