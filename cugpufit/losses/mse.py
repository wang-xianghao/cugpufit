import cunumeric as np

from .loss import Loss

class MSE(Loss):
    def loss_with_residuals(self, outputs, targets):
        residuals = targets - outputs

        return np.mean(np.square(residuals)), residuals