import numpy as np
from task import Task
from .experience_replay import ExperienceReplayMemory
from .actor import Actor
from .critic import Critic
from .ou_noise import OUNoise

class DDPG_Agent:
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_size = task.action_size
        
        # Actor
        self.actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        
        # Critic
        self.critic = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        # Initialize target model parameters
        self.actor_target.model.set_weights(self.actor.model.get_weights())
        self.critic_target.model.set_weights(self.critic.model.get_weights())
        
        # Noise for exploration
        self.mu = 0
        self.sigma = 0.3
        self.theta = 0.15
        self.ounoise = OUNoise(self.action_size, self.mu, self.sigma, self.theta)
        
        # Experience Replay memory
        self.er_memory = ExperienceReplayMemory(capacity=10000, batch_size=64)
        # self.populate_memory(1000)
        
        # RL parameters
        self.gamma = 0.99    # Discount factor
        self.tau = 0.001     # Soft update parameter
        
        
        # Keepng track of learning
        self.learning_rewards = list()
        self.total_reward = None
        self.best_reward = -np.inf
        self.loss = 1
        
        self.batch_size = 64
    
    def reset_episode(self):
        state = self.task.reset()
        self.state = state
        self.ounoise.reset()
        return state
    
    def step(self, action, reward, next_state, done):
        # Add to experience replay memory
        self.er_memory.add((self.state, action, reward, next_state, done))
        
#         experiences = self.er_memory.sample()
#         self.learn(experiences)
        if self.er_memory.len() > self.batch_size:
            self.learn()

        self.state = next_state
    
    def act(self, states):
        action = self.actor.model.predict(np.reshape(states, [-1, self.state_size]))[0]
        step_noise = self.ounoise.sample()
        action = action + step_noise
        return action
        
    def learn(self):
        # Get sample batch from Experience Replay Memory
        states, actions, rewards, next_states, dones = self.er_memory.sample_batch()
        # Convert each list into arrays
        states = np.vstack(states)
        actions = np.array(actions, dtype=np.float32).reshape(-1, self.action_size)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        dones = np.array(dones, dtype=np.uint8).reshape(-1, 1)
        next_states = np.vstack(next_states)
        
        # Get predicted next actions and Q values from target model
        next_actions = self.actor_target.model.predict_on_batch(next_states)
        next_q_values = self.critic_target.model.predict_on_batch([next_states, next_actions])
        
        # Compute Q targets
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        # Train Critic
        self.critic.model.train_on_batch(x=[states, actions], y=q_targets)
        
        # Train Actor
        action_gradients = self.critic.get_action_gradients([states, actions, 0])
        action_gradients = np.reshape(action_gradients[0], (-1, self.action_size))
        self.actor.train_fn([states, action_gradients, 1])
        
        # Soft update target 
        self.soft_update()
  
    def soft_update(self):
        actor_current_weights = np.array(self.actor.model.get_weights())
        critic_current_weights = np.array(self.critic.model.get_weights())
        
        actor_target_weights = np.array(self.actor_target.model.get_weights())
        critic_target_weights = np.array(self.critic_target.model.get_weights())
        
        self.actor_target.model.set_weights(self.tau * actor_current_weights + (1 - self.tau) * actor_target_weights)
        self.critic_target.model.set_weights(self.tau * critic_current_weights + (1 - self.tau) * critic_target_weights)
     
    def populate_memory(self, pretrain_length):
        self.task.reset()
        state, reward, done = self.task.step(self.task.action_sample())
        
        # Make a bunch of random actions and store the experience
        for ii in range(pretrain_length):
            # Make a random action
            action = self.task.action_sample()
            next_state, reward, done = self.task.step(action)
            
            if done:
                # The simulation fails so no next state
                next_state = np.zeros(state.shape)
                self.task.reset()

                # Take one random step to get the 
                state, reward, done = self.task.step(self.task.action_sample())
            else:
                # Add experience to memory
                self.er_memory.add((state, action, reward, next_state))
                state = next_state
          