import cunumeric as np

from cugpufit.fits import GaussNewtonFit
from cugpufit.losses import MeanSquaredError
from cugpufit import Model

class LinearModel(Model):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.W = np.random.randn(self.dimensions, 1)
        self.b = np.zeros(1)
        self.backups = None

    def predict(self, inputs):
        return inputs @ self.W + self.b
    
    def compute_jacobian_with_outputs(self, inputs):
        outputs = inputs @ self.W + self.b
        J = np.empty((inputs.shape[0], self.dimensions + 1))
        J[:, 0:self.dimensions] =  inputs
        J[:, self.dimensions] = 1.0

        return J, outputs

    def update(self, updates):
        self.W += updates[0:self.dimensions].reshape(self.W.shape)
        self.b += updates[self.dimensions].reshape(self.b.shape)

    def backup_parameters(self):
        self.backups = (self.W.copy(), self.b.copy())

    def restore_parameters(self):
        self.W, self.b = map(lambda x: x.copy(), self.backups)
    
    def total_parameters(self):
        return self.W.size() + self.b.size()

if __name__ == '__main__':
    n_samples = 32768
    n_dimensions = 256
    noisy_level = 0.1

    true_weights = np.random.randn(n_dimensions, 1)
    true_bias = np.random.randn(1)
    X_train = np.random.randn(n_samples, n_dimensions)
    Y_train = X_train @ true_weights + noisy_level * np.random.randn(n_samples, 1) + true_bias

    model = LinearModel(n_dimensions)
    gauss_newton = GaussNewtonFit(model, MeanSquaredError())

    # Warmup
    model.compute_jacobian_with_outputs(X_train)

    gauss_newton.fit(X_train, Y_train, 10)