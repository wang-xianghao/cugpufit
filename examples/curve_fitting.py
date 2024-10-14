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
        self.b1 = np.zeros((1, 20), dtype=self.dtype)
        self.W2 = glorot_uniform((20, 1), 21)
        self.b2 = np.zeros((1, 1), dtype=self.dtype)
    
    def predict(self, inputs, train=False):
        # Calculate outputs
        z = np.matmul(self.W1, self.inputs) + self.b1
        a = np.tanh(z)
        y = np.matmul(self.W2, a) + self.b2

        if not train:
            return y
        
        # Calculate jacobian
        tanh_prime = 1 - np.square(a)
        diag_tanh_prime = np.diag(tanh_prime)
        self.jacobian = np.dot(self.W2, np.dot(diag_tanh_prime, self.W1))

        return y

if __name__ == '__main__':
    model = SimpleDense(MSE, dtype=np.float64)

    print(MSE().loss_with_residuals(np.zeros(10), np.random.uniform(-1, 1, 10)))