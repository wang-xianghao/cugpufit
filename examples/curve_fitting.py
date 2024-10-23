import cunumeric as np
import matplotlib.pyplot as plt

from itertools import accumulate

from cugpufit.fits import LevenbergMarquardtFit
from cugpufit.losses import MeanSquaredError
from cugpufit import Model

class CurveModel(Model):
    def __init__(self, latent_units, dtype: np.dtype=np.float64, seed: int=0):
        self.dtype = dtype
        self.__rng = np.random.default_rng(seed)
        
        def glorot_uniform(units, shape):
            limit = np.sqrt(6.0 / units)
            return self.__rng.uniform(-limit, limit, size=shape, dtype=self.dtype)
        
        self.latent_units = latent_units
        self.W1 = glorot_uniform(21, (1, self.latent_units))
        self.b1 = np.zeros(self.latent_units, dtype=self.dtype)
        self.W2 = glorot_uniform(21, (self.latent_units, 1))
        self.b2 = np.zeros(1, dtype=self.dtype)
        
        self.backups = None
        self.paramters = [self.W1, self.b1, self.W2, self.b2]
        self.paramters_sizes = list(map(lambda x: x.size, self.paramters))
        self.num_parameters = sum(self.paramters_sizes)
        self.paramters_offsets = list(accumulate(self.paramters_sizes))

    def predict(self, inputs):
        return np.tanh(inputs @ self.W1 + self.b1) @ self.W2 + self.b2
    
    def __view_paramter(self, i, data):
        return data[...,
                    self.paramters_offsets[i] - self.paramters_sizes[i]:self.paramters_offsets[i]]
    
    def compute_jacobian_with_outputs(self, inputs):
        z = np.tanh(inputs @ self.W1 + self.b1)
        outputs = z @ self.W2 + self.b2
        
        J = np.empty((inputs.shape[0], self.num_parameters), dtype=self.dtype)
        
        # Compute jacobians
        # b1 jacobian
        J_b1 = self.__view_paramter(1, J)[...] = np.multiply(1 - np.square(z), self.W2.T)
        # W1 jacobian
        self.__view_paramter(0, J)[...] = (inputs[:, :, np.newaxis] @ 
                                            J_b1[:, np.newaxis, :]).reshape(inputs.shape[0], self.latent_units)
        # W2 jacobian
        self.__view_paramter(2, J)[...] = z
        # b2 jacobian
        self.__view_paramter(3, J)[...] = 1.0

        return J, outputs

    def update(self, updates):
        for i, p in enumerate(self.paramters):
            p += self.__view_paramter(i, updates).reshape(p.shape)

    def backup_parameters(self):
        self.backups = (self.W1.copy(), self.b1.copy(),
                        self.W2.copy(), self.b2.copy())

    def restore_parameters(self):
        self.W1, self.b1, self.W2, self.b2 = map(lambda x : x.copy(), self.backups)
        self.paramters = [self.W1, self.b1, self.W2, self.b2]
        
    def total_parameters(self):
        return self.num_parameters
        
if __name__ == '__main__':
    n_samples = 20000
    batch_size = 1000
    latent_units = 20
    data_type = np.float64

    # Prepare train data
    X_train = np.linspace(-1, 1, n_samples, dtype=data_type).reshape(n_samples, 1)
    Y_train = np.sinc(10 * X_train).reshape(n_samples, 1)
    order = np.random.permutation(n_samples)
    X_train = X_train[order, :]
    Y_train = Y_train[order, :]

    model = CurveModel(latent_units, dtype=data_type)
    lm = LevenbergMarquardtFit(model, MeanSquaredError())

    # Warmup
    model.compute_jacobian_with_outputs(X_train)

    # Train
    lm.fit(X_train, Y_train, epoches=100, batch_size=batch_size, metrics=['loss',
                                                                         'attempts',
                                                                         'singulars',
                                                                         'invalid_updates'])

    # Draw results
    X_test = np.linspace(-1, 1, n_samples, dtype=data_type).reshape(n_samples, 1)
    Y_test = np.sinc(10 * X_test).reshape(n_samples, 1)
    
    plt.plot(X_test, Y_test, 'b-', label="reference")
    plt.plot(X_test, model.predict(X_test), 'r--', label="lm")
    plt.savefig('curve_fitting.png')