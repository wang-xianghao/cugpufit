from abc import ABC, abstractmethod

class Loss:
    @abstractmethod
    def residuals(self, targets, outputs):
        pass
    
    @abstractmethod
    def __call__(self, targets, outputs):
        pass