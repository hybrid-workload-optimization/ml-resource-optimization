
class ModelFeature(object):
    
    def __init__(self, window):
        self.input_width = 10
        self.label_width = 10
        self.column_indices = [1, 2]
        self.label_columns = [1, 2]
        self.__call__(window)
            
    def __call__(self, window):
        if window is not None:
            self.input_width = window.input_width
            self.label_width = window.label_width
            self.column_indices = window.column_indices
            self.label_columns = window.label_columns