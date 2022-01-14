import inspect

def isFunction(module, func):
    if hasattr(module, func):
        for method in inspect.getmembers(module, predicate=inspect.ismethod):
            method_name, method_obj = method
            if method_name == func:
                return True
    return False


def method_exists(instance, method):
    return hasattr(instance, method) and inspect.ismethod(getattr(instance, method))