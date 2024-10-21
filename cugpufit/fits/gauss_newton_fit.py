import cunumeric as np
from legate.timing import time as letime

from .fit import Fit
from cugpufit.models.model import Model
from cugpufit.losses.loss import Loss

class GaussNewtonFit(Fit):
    def __init__(self, model: Model, loss: Loss):
        super().__init__(model, loss)
    
    def fit_step(self, inputs, targets):        
        J, outputs = self.model.compute_jacobian_with_outputs(inputs)
        residuals = self.loss.residuals(outputs, targets)
        JJ = np.matmul(J.T, J)
        
        rhs = np.matmul(J.T, residuals)
        updates = np.linalg.solve(JJ, rhs)

        loss_val = self.loss(outputs, targets)
            
        return np.squeeze(updates), loss_val
    
    def fit(self, inputs, targets, epoches, batch_size=-1, verbose=True):
        if batch_size == -1:
            batch_size = inputs.shape[0]

        n_batches = inputs.shape[0] // batch_size
        assert n_batches * batch_size == inputs.shape[0]

        inputs_all = np.split(inputs, n_batches, axis=0)
        targets_all = np.split(targets, n_batches, axis=0)

        for epoch in range(1, epoches + 1):
            if verbose:
                t_start = letime()
                
            for (inputs_batch, targets_batch) in zip(inputs_all, targets_all):
                updates, loss_val = self.fit_step(inputs_batch, targets_batch)
                self.model.update(updates)
                
            if verbose:
                t_end = letime()
            
            if verbose:
                t_batch = (t_end - t_start) / (1e3 * n_batches)
                
            if verbose:
                print(f'epoch {epoch}/{epoches} loss: {loss_val}, avg. batch time: {t_batch} ms')