import cunumeric as np

from cugpufit.model import Model
from cugpufit.losses.mse import MSE

class SimpleDense(Model):
    def __init__(self, estimator, dtype):
        super().__init__('curve fitting', estimator, dtype)

        def glorot_uniform(shape, k):
            limit = np.sqrt(6 / k)
            return np.random.uniform(-limit, limit, shape, dtype=self.dtype)

        self.W1 = glorot_uniform((1, 20), 21)
        self.b1 = np.zeros(20, dtype=self.dtype)
        self.W2 = glorot_uniform((20, 1), 21)
        self.b2 = np.zeros(1, dtype=self.dtype)

        self.parameters = [self.W1, self.b1, self.W2, self.b2]
    
    def predict(self, inputs, train=False):
        N, D = inputs.shape

        # Calculate outputs
        y1 = np.tanh(inputs @ self.W1 + self.b1)
        y2 = y1 @ self.W2 + self.b2

        if not train:
            return y2

        # TODO: Calculate jacobian by manual backpropagation
        self.jacobian = None

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