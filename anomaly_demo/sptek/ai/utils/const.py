class _constant2:
            
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise Exception('Cannot assign value to variable.')
        self.__dict__[name] = value

    def __delattr__(self, name):
        if name in self.__dict__:
            raise Exception('Cannot delete variable.')