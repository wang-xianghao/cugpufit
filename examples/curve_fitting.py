import cunumeric as np
# import numpy as np

from itertools import accumulate

from cugpufit.model import Model
from cugpufit.losses.mse import MSE

# TODO: data types in the model should be consistent
class SimpleDense(Model):
    def __init__(self, estimator, dtype):
        super().__init__('curve fitting', estimator, dtype)

        def glorot_uniform(shape, k):
            limit = np.sqrt(6 / k)
            return np.random.uniform(-limit, limit, shape)

        self.W1 = glorot_uniform((1, 20), 21)
        self.b1 = np.zeros(20, dtype=self.dtype)
        self.W2 = glorot_uniform((20, 1), 21)
        self.b2 = np.zeros(1, dtype=self.dtype)

        self.parameters = [self.W1, self.b1, self.W2, self.b2]
        self.n_parameters = sum(map(np.size, self.parameters))
        self.offsets = list(accumulate(map(np.size, self.parameters)))
    
    def predict(self, inputs, train=False):
        N, D = inputs.shape

        # Calculate outputs
        y1 = np.tanh(inputs @ self.W1 + self.b1)
        y2 = y1 @ self.W2 + self.b2

        if not train:
            return y2

        # Calculate jacobian manually
        # TODO: use autograd tools
        self.jacobian = np.empty((N, self.n_parameters), dtype=model.dtype)
        
        for i in range(N):
            self.jacobian[i:i+1, self.offsets[0]:self.offsets[1]] = y1[i, :] * self.W2.T
            self.jacobian[i:i+1, 0:self.offsets[0]] = np.dot(inputs[i, :], self.jacobian[i:i+1, self.offsets[0]:self.offsets[1]])
            self.jacobian[i:i+1, self.offsets[1]:self.offsets[2]] = y1[i, :]
            self.jacobian[i:i+1, self.offsets[2]:self.offsets[3]] = 1.0

        return y2

    def update(self, updates):
        offset = 0
        for p in self.parameters:
            p += updates[offset:offset + p.size].reshape(p.shape)
            offset += p.size

if __name__ == '__main__':
    model = SimpleDense(MSE, dtype=np.float64)

    model.predict(np.random.rand(100, 1), train=True)
    print(model.jacobian)