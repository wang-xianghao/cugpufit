from abc import ABC, abstractmethod

class Damping(ABC):
    def __init__(self, starting_value=1e-3, 
                 dec_factor=0.1, inc_factor=10.0,
                 min_value=1e-10, max_value=1e+10):
        self.starting_value = starting_value
        self.dec_factor = dec_factor
        self.inc_factor = inc_factor
        self.min_value = min_value
        self.max_value = max_value
        
    @abstractmethod
    def init_step(self, damping_factor, loss):
        return damping_factor
        