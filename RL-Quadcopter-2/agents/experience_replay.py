from collections import deque
import numpy as np

class ExperienceReplayMemory:
    def __init__(self, capacity, batch_size):
        self.batch_size = batch_size
        self.mem = deque(maxlen=capacity)
        
    def add(self, env_reaction):
        #print('add to memory: {}'.format(env_reaction))
        self.mem.append(env_reaction)
        
    def sample_batch(self):
        indexes = np.random.choice(a=np.arange(len(self.mem)), size=self.batch_size, replace=False)
        # if debug: print(indexes)
        states = list()
        actions = list()
        rewards = list()
        dones = list()
        next_states = list()
        for index in indexes:
            if self.mem[index] is None:
                print(self.mem[index])
            st, at, rt, st_1, dt = self.mem[index]
            states.append(st)
            actions.append(at)
            rewards.append(rt)
            dones.append(dt)
            next_states.append(st_1)
        return states, actions, rewards, next_states, dones
    
    def len(self):
        return len(self.mem)