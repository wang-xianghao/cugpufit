from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, model_name, estimator):
        self.model_name = model_name
        self.estimator = estimator
        self._jacobian = None

    @abstractmethod
    def predict(self, inputs, train):
        pass

    @property
    def jacobian(self):
        if self._jacobian == None:
            raise ValueError('Please call predict() with train before querying jacobian.')

        return self._jacobian