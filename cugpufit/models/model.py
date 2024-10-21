from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def predict(self, inputs):
        pass
    
    @abstractmethod
    def compute_jacobian_with_outputs(self, inputs):
        pass
    
    @abstractmethod
    def update(self, updates):
        pass
    
    @abstractmethod
    def push_parameters(self):
        pass
    
    @abstractmethod
    def pop_parameters(self):
        pass