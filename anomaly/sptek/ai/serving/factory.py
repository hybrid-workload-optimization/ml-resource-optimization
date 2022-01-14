
from .data_serving import ForecastAdaptor, TrainAdaptor

class DataAdaptorFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master):
        
        service_type = master.service_type()
        db = master.database()

        # service type is forecast
        adaptors = []
        if service_type == 'forecast':
            for db_type in db.type:
                adaptors.append(ForecastAdaptor(master, db_type))
            return (service_type, adaptors)
        
        # service type is train
        if service_type == 'train':
            for db_type in db.type:
                adaptors.append(TrainAdaptor(master, db_type))
            return (service_type, adaptors)
        
        # service type is auto
        if service_type == 'auto':
            raise NotImplementedError("This function is not supported. (service_type = 'auto')")
    