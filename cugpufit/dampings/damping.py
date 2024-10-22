from abc import ABC, abstractmethod

class Damping(ABC):
    def __init__(self, starting_value, dec_factor, inc_factor, min_value, max_value):
        self.starting_value = starting_value
        self.dec_factor = dec_factor
        self.inc_factor = inc_factor
        self.min_value = min_value
        self.max_value = max_value
        
    @abstractmethod
    def init_step(self, damping_factor, loss):
        pass
    
    @abstractmethod
    def decrease(self, damping_factor, loss):
        pass
    
    @abstractmethod
    def increase(self, damping_factor, loss):
        pass 
    
    @abstractmethod
    def stop_training(self, damping_factor, loss):
        pass
    
    @abstractmethod
    def apply(self, damping_factor, JJ):
        pass