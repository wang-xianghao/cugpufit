import cunumeric as np
from abc import ABC, abstractmethod

from cugpufit import Model, Loss

class Fit(ABC):
    def __init__(self, model: Model, loss: Loss):
        self.model = model
        self.loss = loss
        
    @abstractmethod
    def fit(self, inputs, targets):
        pass