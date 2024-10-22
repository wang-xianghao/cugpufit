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
        
        # Fit with damping
        loss = self.loss(targets, outputs)
        singulars = 0 # Count singular Hessian's occurrences
        invalid_updates = 0 # Count invalid updates
        damping_factor = self.damping_algorithm.init_step(self.damping_factor, 
                                                          JJ)
        attempts = 0
        while attempts < self.attempts_per_step:
            attempts += 1
            update_computed = False
            try:
                # Apply the damping to the gauss-newton hessian approximation.
                JJ_damped = self.damping_algorithm.apply(damping_factor, JJ)

                # Compute the updates:
                # overdetermined: updates = (J'*J + damping)^-1*J'*residuals
                # underdetermined: updates = J'*(J*J' + damping)^-1*residuals
                updates = self.compute_gauss_newton(J, JJ_damped, rhs)
            except Exception as e: # Probably due to singular Hessian
                singulars += 1
                del e
            else:
                # Ensure all updates are invalid
                if np.all(np.isfinite(updates)):
                    update_computed = True
                    self.model.update(updates)
                else:
                    invalid_updates += 1
            
            if update_computed:
                # Check whether updated model performs better; if not, restore old parameters
                outputs = self.model.predict(inputs)
                new_loss = self.loss(targets, outputs)
                
                if new_loss < loss:
                    loss = new_loss
                    damping_factor = self.damping_algorithm.decrease(damping_factor,
                                                                     loss)
                    self.model.backup_parameters()
                    break
                
                self.model.restore_parameters()
            
            # Increase damping factor to prefer gradient descent
            damping_factor = self.damping_algorithm.increase(damping_factor, loss)
            stop_training = self.damping_algorithm.stop_training(damping_factor,
                                                                 loss)
            if stop_training:
                break
        
        self.damping_factor = damping_factor
        
        return {'loss': loss,
                'attempts': attempts,
                'singulars': singulars,
                'invalid_updates': invalid_updates}
    
    def __epoch_start(self):
        self.metrics_data = {
            'loss': 0.0,
            'attempts': 0,
            'singulars': 0,
            'invalid_updates': 0
        }
        self.elapsed_ms = 0.0
        
    def __epoch_end(self):
        print(f'epoch {self.__epoch}/{self.__epoches} finished in {self.elapsed_ms} ms')
        for name in self.metrics:
            data = self.metrics_data[name]
            print(f'\t{name}: {data}')
        print()
    
    def __batch_start(self):
        self.__start_time = letime()
        
    def __batch_end(self, results):
        self.elapsed_ms += (letime() - self.__start_time) / 1e3
        for name in self.metrics:
            self.metrics_data[name] += results[name]
    
    def fit(self, inputs, targets, epoches, batch_size=-1, metrics: list[str]=['loss']):
        self.model.backup_parameters()
        
        if batch_size == -1:
            batch_size = inputs.shape[0]

        self.metrics = metrics
        self.__epoches = epoches
        
        # Split data into batches
        n_batches = inputs.shape[0] // batch_size
        assert n_batches * batch_size == inputs.shape[0]
        inputs_all = np.split(inputs, n_batches, axis=0)
        targets_all = np.split(targets, n_batches, axis=0)

        for epoch in range(1, epoches + 1):
            self.__epoch = epoch
            self.__epoch_start()        
            for (inputs_batch, targets_batch) in zip(inputs_all, targets_all):
                self.__batch_start()
                results = self.fit_step(inputs_batch, targets_batch)
                self.__batch_end(results)
            self.__epoch_end()