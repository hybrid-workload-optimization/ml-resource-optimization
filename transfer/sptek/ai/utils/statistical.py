from . import logger

def normalize(data, mean, std, inverse=False):
    if inverse is True :
        return data * std + mean
    return (data - mean) / std