from .algorithm import ModelHandler
from .bilstm import BiLSTM
from .lstm import LSTM
from .word2vec import EmbeddingVector
from ... import prefix
from ...master import Master
from ...serving import ImportAdaptor
from ...serving import ModelServing
from ...utils import logger

class ModelFactory(object):

    def __init__(self):
        pass

    @classmethod
    def create(cls, master, searcher=None, feature=None, action='import'):
        if action == 'import':
            meta = master.import_model()
            if meta.type == "embed":
                return EmbeddingVector(meta, searcher)
                
        if action == 'train':
            meta = master.machine(dtype='sequential')
            # logger.debug(f"{meta}")
            
            hyper = master.machine(dtype='hyperparameter')
            # logger.debug(f"{hyper}")
            
            stopping = master.machine(dtype='earlystopping')
            # logger.debug(f"{stopping}")
            
            if meta.type == 'sequential':
                if meta.algorithm == 'BiLSTM':
                    model = BiLSTM (
                        meta.unit,
                        feature.input_width,
                        feature.label_width,
                        len(feature.column_indices),
                        len(feature.label_columns),
                        hyper.epochs,
                        hyper.loss,
                        hyper.optimizer,
                        hyper.metrics,
                        stopping
                    )
                    model.build()
                    return model
                
                if meta.algorithm == 'LSTM':
                    model = LSTM (
                        meta.unit,
                        feature.input_width,
                        feature.label_width,
                        len(feature.column_indices),
                        len(feature.label_columns),
                        hyper.epochs,
                        hyper.loss,
                        hyper.optimizer,
                        hyper.metrics,
                        stopping
                    )
                    model.build()
                    return model
                
        if action == 'transfer':
            meta = master.machine(dtype='sequential')
            # logger.debug(f"{meta}")
            
            hyper = master.machine(dtype='hyperparameter')
            # logger.debug(f"{hyper}")
            
            stopping = master.machine(dtype='earlystopping')
            # logger.debug(f"{stopping}")
            
            if meta.algorithm == 'BiLSTM':
                model = BiLSTM (units=None, inputs=None, outputs=None, in_features=None, out_features=None,
                                epochs=hyper.epochs, loss=hyper.loss, optimizer=hyper.optimizer, metrics=hyper.metrics, earlystopping=stopping)
                model._is_transfer_ = meta.transfer
                return model
            
        if action == 'forecast':
            meta = master.machine(dtype='sequential')
            # logger.debug(f"{meta}")
            
            if meta.algorithm == 'BiLSTM':
                    return BiLSTM (units=None, inputs=None, outputs=None, in_features=None, out_features=None,
                                   epochs=None, loss=None, optimizer=None, metrics=None, earlystopping=None)
    
    @classmethod
    def hyperparameter(cls):
        master = Master()
        hyper = master.machine(dtype='hyperparameter')
        # logger.debug(f"{hyper}")
        return hyper
    
    @classmethod
    def load(cls, path, action='import'):
        if action == 'forecast':
            file = prefix.model.name(path)
            adaptor = ImportAdaptor('keras', ModelServing())
            return adaptor.get(file)