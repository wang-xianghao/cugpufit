import cunumeric as np

from .loss import Loss

class MeanSquaredError(Loss):
    def residuals(self, targets, outputs):
        return targets - outputs
    
    def __call__(self, targets, outputs):
        return np.mean(np.square(targets - outputs))