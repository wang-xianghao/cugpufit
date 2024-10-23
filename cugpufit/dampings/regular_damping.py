import cunumeric as np

from .damping import Damping

# TODO: add adaptive scaling
class RegularDamping(Damping):
    def __init__(self, starting_value=0.001,
                 dec_factor=0.1, inc_factor=10,
                 min_value=1e-10, max_value=10000000000):
        super().__init__(starting_value, dec_factor, inc_factor, min_value, max_value)
        
    def init_step(self, damping_factor, loss):
        return damping_factor
    
    def decrease(self, damping_factor, loss):
        return max(
            damping_factor * self.dec_factor,
            self.min_value)
    
    def increase(self, damping_factor, loss):
        return min(
            damping_factor * self.inc_factor,
            self.max_value)

    def stop_training(self, damping_factor, loss):
        return damping_factor >= self.max_value
    
    def apply(self, damping_factor, JJ):
        damping = np.multiply(damping_factor, np.eye(JJ.shape[0], dtype=JJ.dtype), dtype=JJ.dtype)
        return damping + JJ
    