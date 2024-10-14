from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, model_name, estimator, dtype):
        self.model_name = model_name
        self.estimator = estimator
        self.jacobian = None
        self.dtype = dtype

    @abstractmethod
    def predict(self, inputs, train):
        pass