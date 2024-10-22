import cunumeric as np
from legate.timing import time as letime

from .fit import Fit
from cugpufit.models.model import Model
from cugpufit.losses.loss import Loss
from cugpufit.dampings.damping import Damping
from cugpufit.dampings import RegularDamping

class LevenbergMarquardtFit(Fit):
    def __init__(self, model: Model, loss: Loss,
                 damping_algorithm: Damping=RegularDamping(),
                 attempts_per_step: int=10):
        super().__init__(model, loss)
        
        self.damping_algorithm = damping_algorithm
        self.attempts_per_step = attempts_per_step
        
        self.damping_factor = self.damping_algorithm.starting_value
        
        # TODO: optimization on gauss-newton
        # self.init_gauss_newton = None
        # self.compute_gauss_newton = None
        self.init_gauss_newton = self.__init_gauss_newton_underdetermined
        self.compute_gauss_newton = self.__compute_gauss_newton_underdetermined
    
    # TODO: add gauss-newton overdetermined
    # def __init_gauss_newton_overdetermined(self, inputs, targets):
    #     J, outputs = self.model.compute_jacobian_with_outputs(inputs)
    #     residuals = self.loss(outputs, targets)
        
    #     JJ = J.T @ J
    
    def __init_gauss_newton_underdetermined(self, inputs, targets):
        J, outputs = self.model.compute_jacobian_with_outputs(inputs)
        residuals = self.loss.residuals(targets, outputs)
        
        JJ = J @ J.T
        rhs = residuals
        
        return J, JJ, rhs, outputs
    
    def __compute_gauss_newton_underdetermined(self, J, JJ, rhs):
        g = np.linalg.solve(JJ, rhs)
        updates = np.matmul(J.T, g)
        
        return np.squeeze(updates)
    
    def fit_step(self, inputs, targets):
        J, JJ, rhs, outputs = self.init_gauss_newton(inputs, targets)
        
        batch_size = inputs.shape[0]
        normalization_factor = 1.0 / batch_size
        
        # Normalization
        np.multiply(normalization_factor, JJ, out=JJ)
        np.multiply(normalization_factor, rhs, out=rhs)
        
        loss = self.loss(targets, outputs)
        
        stop_training = False
        attempt = 0
        damping_factor = self.damping_algorithm.init_step(
            self.damping_factor, loss)

        while True:
            update_computed = False
            try:
                # Apply the damping to the gauss-newton hessian approximation.
                JJ_damped = self.damping_algorithm.apply(damping_factor, JJ)

                # Compute the updates:
                # overdetermined: updates = (J'*J + damping)^-1*J'*residuals
                # underdetermined: updates = J'*(J*J' + damping)^-1*residuals
                updates = self.compute_gauss_newton(J, JJ_damped, rhs)
            except Exception as e:
                print(f'Encountered singular Hessian: {e}')
                del e
            else:                
                if np.all(np.isfinite(updates)):
                    update_computed = True
                    self.model.update(updates)
                                
            if attempt < self.attempts_per_step:
                attempt += 1
                
                if update_computed:
                    outputs = self.model.predict(inputs)
                    new_loss = self.loss(targets, outputs)
                    
                    if new_loss < loss:
                        loss = new_loss
                        damping_factor = self.damping_algorithm.decrease(
                            damping_factor, loss)
                        self.model.backup_parameters()
                        break
                    
                    self.model.restore_parameters()
                
                damping_factor = self.damping_algorithm.increase(damping_factor, loss)
                
                stop_training = self.damping_algorithm.stop_training(
                    damping_factor, loss)
                
                if stop_training:
                    break
            else:
                break
        self.damping_factor = damping_factor
                
        return loss, attempt, damping_factor
    
    def fit(self, inputs, targets, epoches, batch_size=-1, verbose=True):
        self.model.backup_parameters()
        
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
                loss_val, attemp, damping_factor = self.fit_step(inputs_batch, targets_batch)
                
            if verbose:
                t_end = letime()
            
            if verbose:
                t_batch = (t_end - t_start) / (1e3 * n_batches)
                
            if verbose:
                print(f'epoch {epoch}/{epoches} loss: {loss_val}, attemps: {attemp}, damping_factor: {damping_factor}, avg. batch time: {t_batch} ms')