import cunumeric as np

from .loss import Loss

class MeanSquaredError(Loss):
    def residuals(self, outputs, targets):
        return targets - outputs
    
    def __call__(self, outputs, targets):
        return np.mean(np.square(targets - outputs))