import numpy as np
from task import Task
from memory import Memory
import tensorflow as tf
from qnetwork import QNetwork
from .actor import Actor
from .critic import Critic
from .ou_noise import OUNoise
from .experience_replay import ExperienceReplayBuffer

class DDPG_Agent:
    def __init__(self, task, noise, memory, rl_param, nn_hidden, actor_lr, critic_lr, q_lambda):
        self.task = task
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.state_space = task.state_size
        self.action_space = task.action_size
        self.q_lambda = q_lambda
        
        self.actor = Actor(self.state_space, self.action_space, self.action_low, self.action_high, hidden_units=nn_hidden[0], learning_rate=actor_lr, q_lambda=q_lambda)
        self.actor_target = Actor(self.state_space, self.action_space, self.action_low, self.action_high, hidden_units=nn_hidden[0], learning_rate=actor_lr, q_lambda=q_lambda)
        
        self.critic = Critic(self.state_space, self.action_space, hidden_units=nn_hidden[1], learning_rate=critic_lr, q_lambda=q_lambda)
        self.critic_target = Critic(self.state_space, self.action_space, hidden_units=nn_hidden[1], learning_rate=critic_lr, q_lambda=q_lambda)
        
        self.actor_target.model.set_weights(self.actor.model.get_weights())
        self.critic_target.model.set_weights(self.critic.model.get_weights())
        
        # Noise for exploration
        self.mean = noise[0]
        self.sigma = noise[1]
        self.theta = noise[2]
        self.ounoise = OUNoise(self.action_space, self.mean, self.sigma, self.theta)
        
        # Experience Replay memory
        self.capacity = memory[0]
        self.batch_size = memory[1]
        self.er_buffer = ExperienceReplayBuffer(capacity=self.capacity, batch_size=self.batch_size)
        
        # RL parameters
        self.gamma = rl_param[0]
        self.t = rl_param[1]
        
        
        # Keepng track of learning
        self.learning_rewards = list()
        self.total_reward = None
        self.best_reward = -np.inf
        self.losses = list()
    
    def restart_task(self):
        if self.total_reward is not None:
            self.learning_rewards.append(self.total_reward)
            if self.total_reward > self.best_reward: self.best_reward = self.total_reward
        self.total_reward = 0
        state = self.task.reset()
        self.state = state
        self.ounoise.restart()
        return state
    
    
    def act(self, state, epsilon):
        self.action_wo_noise = self.actor.model.predict(np.reshape(state, newshape=(-1, self.state_space)))
        self.step_noise = self.ounoise.sample()*epsilon
        action = np.array(self.action_wo_noise[0] + self.step_noise[0]).reshape(-1, self.action_space)
        action_clipped = np.clip(a=action, a_min=self.action_low, a_max = self.action_high)
        return action_clipped
    
    def store_learn(self, state, action, reward, done, next_state):
        self.er_buffer.add_env_reaction((state, action, reward, done, next_state))
        
        if len(self.er_buffer.mem) > self.batch_size:
            self.learn()
            
        self.total_reward += reward
        
        self.state = next_state
        
    def soft_update(self):
        actor_current = np.array(self.actor.model.get_weights())
        critic_current = np.array(self.critic.model.get_weights())
        
        actor_target = np.array(self.actor_target.model.get_weights())
        critic_target = np.array(self.critic_target.model.get_weights())
        
        self.actor_target.model.set_weights(actor_target * (1 - self.t) + self.t * actor_current)
        self.critic_target.model.set_weights(critic_target * (1 - self.t) + self.t * critic_current)
        
    def learn(self):
        states, actions, rewards, dones, next_states = self.er_buffer.sample_batch()
        states = np.vstack(states)
        actions = np.array(actions, dtype=np.float32).reshape(-1, self.action_space)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        dones = np.array(dones, dtype=np.uint8).reshape(-1, 1)
        next_states = np.vstack(next_states)
        
        next_actions = self.actor_target.model.predict_on_batch(next_states)
        next_q_values = self.critic_target.model.predict_on_batch([next_states, next_actions])
        
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.critic.model.train_on_batch(x=[states, actions], y=targets)
        self.losses.append(loss)
        
        action_gradients = self.critic.get_action_gradients([states, actions, 0])
        action_gradients_prev = action_gradients
        action_gradients = np.reshape(action_gradients[0], (-1, self.action_space))
        
        self.actor.train_fn([states, action_gradients, 1])
        
        self.soft_update()
        
            
           
            
            
class DeepQ_Agent:
    def __init__(self, task, memory_size=10000, pretrain_length=20):
        # Task (Environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        
        self.w = np.random.normal(
            size=(self.state_size, self.action_size),   # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size)))  # start producing actions in a decent range
            
        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1
        
        # Experience Memory
        self.memory = Memory(memory_size)
        self.pretrain_length = pretrain_length
        
        
        # Q-network
        self.hidden_size = 64
        self.learning_rate = 0.0001
        tf.reset_default_graph()
        self.mainQN = QNetwork(name='main', state_size=self.state_size, action_size=self.action_size, hidden_size=self.hidden_size, learning_rate=self.learning_rate)
        
        # Episode variables
        self.reset_episode()
    
    def populate_memory(self):
        self.task.reset()
        state, reward, done = self.task.step(self.task.action_sample())
        
        # Make a bunch of random actions and store the experience
        for ii in range(self.pretrain_length):
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
                self.memory.add((state, action, reward, next_state))
                state = next_state
                
        return state
                
    def train(self):
        state = self.populate_memory()
        
        # Train with experiences
        saver = tf.train.Saver()
        rewards_list = []
        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            
            step = 0
            train_episodes = 10
            max_steps = 200
            gamma = 0.99
            
            # Exploration parameters
            explore_start = 1.0
            explore_stop = 0.01
            decay_rate = 0.0001
            
            batch_size = 20
            
            for ep in range(1, train_episodes):
                total_reward = 0
                t = 0
                while t < max_steps:
                    step += 1
                    
                    # Explore or Exploit
                    explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate*step)
                    if explore_p > np.random.rand():
                        # Make a random action
                        action = self.task.action_sample()
                    else:
                        # Get action from Q-network
                        feed = {self.mainQN.inputs_: state.reshape((1, *state.shape))}
                        Qs = sess.run(self.mainQN.output, feed_dict=feed)
                        action = self.mainQN.output
                        print('action: {}'.format(action))
                   
                    # Take action, get new state and reward
                    next_state, reward, done = self.task.step(action)
                    
                    self.total_reward += reward
                    
                    if done:
                        # the episode ends so no next state
                        next_state = np.zeros(state.shape)
                        t = max_steps
                        
                        print('Episode: {}'.format(ep),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_p))
                        rewards_list.append((ep, total_reward))
                        
                        # Add experience to memory
                        self.memory.add((state, action, reward, next_state))
                        
                        # start new episode
                        self.task.reset()
                        
                        # Take one random step
                        state, reward, done = self.task.step(self.task.action_sample())
                        
                    else:
                        # Add experience to memory
                        self.memory.add((state, action, reward, next_state))
                        state = next_state
                        t += 1
                        
                    # Sample mini-batch from memory
                    batch = self.memory.sample(batch_size)
                    states = np.array([each[0] for each in batch])
                    actions = np.array([each[1] for each in batch])
                    rewards = np.array([each[2] for each in batch])
                    next_states = np.array([each[3] for each in batch])
                    
                    # Train network
                    target_Qs = sess.run(self.mainQN.output, feed_dict={self.mainQN.inputs_:next_states})
                    
                    # Set target_Qs to 0 for states where episode ends
                    episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                    target_Qs[episode_ends] = (0, 0, 0, 0)
                    
                    targets = rewards + gamma * np.max(target_Qs, axis=1)
                    
                    loss, _ = sess.run([self.mainQN.loss, self.mainQN.opt],
                                       feed_dict={self.mainQN.inputs_: states,
                                                  self.mainQN.targetQs_: targets,
                                                  self.mainQN.actions_: actions})
                    
            saver.save(sess, "checkpoints/quadcopter.ckpt")
                        
                        
        
    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state
    
    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1
        
        # Learn, if at end of episode
        if done:
            self.learn()
            
    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        return action
    
    
    def learn(self):
        # Learn by fandom policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # euqal noise in all directions
        
    