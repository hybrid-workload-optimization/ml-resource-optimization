
import os
from pathlib import Path   # This demo requires pip install for Python < 3.4
import sys
try:
    from urllib.parse import urlparse, unquote
    from urllib.request import url2pathname
except ImportError:  # backwards compatability:
    from urlparse import urlparse
    from urllib import unquote, url2pathname

def uri(path):
    return Path(path).as_uri()

def uri_to_normpath(uri):
    parsed = urlparse(uri)
    host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
    return os.path.normpath(
        os.path.join(host, url2pathname(unquote(parsed.path)))
    )

def uri_to_absolutized(uri):
    parsed = urlparse(uri)
    host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
    path = os.path.join(host, url2pathname(unquote(os.path.abspath( parsed.path ))))
    return path

def absolutized(path):
    return os.path.abspath(path=path)