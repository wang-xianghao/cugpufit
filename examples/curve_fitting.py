from cugpufit.model import Model


class SimpleDense(Model):
    def __init__(self, estimator):
        super().__init__('curve fitting', estimator)
    
    def predict(self, inputs, train=False):
        pass


if __name__ == '__main__':
    model = SimpleDense(None)
