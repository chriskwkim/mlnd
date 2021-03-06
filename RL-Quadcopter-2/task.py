import numpy as np
import math
import random
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., action_repeat=3, target_pos=None, debug=False):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = int(action_repeat)

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.debug=debug

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
#         reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = np.tanh(1. - 0.003 * (np.sqrt(np.square(self.sim.pose[:3] - self.target_pos).sum())))
#         reward = 1. - np.tanh(np.sqrt(np.square(self.sim.pose[:3] - self.target_pos).sum()))
#         bonus = 1. - np.tanh(self.target_pos[2] - self.sim.pose[2]) 
#         reward += bonus
#         reward = reward / 2.0
        # reward = 1 / (1 + math.exp(-reward))
        if self.sim.done and self.sim.runtime > self.sim.time:
            reward = -1.0
        
        # reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        
        if self.debug:
            print('------ Step --------')
            print('Position', self.sim.pose[:3], 'Z Speed', self.sim.v[2])
            print()
            print('Position reward', reward_position)
            print('Position Z reward', reward_z_axis)
            #print('Speed Z reward', reward_speed_z)
            print('Final Reward', reward)
            
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    