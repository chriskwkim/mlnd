import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process"""
    def __init__(self, size, mu, sigma, theta):
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.size) * self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        
        self.state = x + dx
        return self.state
        