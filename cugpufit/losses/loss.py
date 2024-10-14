from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def loss_with_residuals(self, outputs, targets):
        pass