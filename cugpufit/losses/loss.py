from abc import ABC, abstractmethod

class Loss:
    @abstractmethod
    def residuals(self, outputs, targets):
        pass
    
    @abstractmethod
    def __call__(self, outputs, targets):
        pass