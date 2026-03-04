# ============ IMPORTS ============
import numpy as np
import gymnasium as gym
import random
from collections import Counter
import numpy.matlib 
from itertools import combinations


# Note : Gymnasium is an open-source Python library, 
# maintained by the Farama Foundation 
# (it is the successor to the original OpenAI Gym library), 
# that provides a standard API and a diverse collection of 
# simulated environments for developing and comparing 
# reinforcement learning (RL) algorithms.

"""
read: https://gymnasium.farama.org/introduction/create_custom_env/
"""

# =================================


# ================ OPENAI GYM CLASS DEFINATIONS ================
class MimoEnv(gym.Env):

    def __init__(self, H, se_max):
        super(MimoEnv, self).__init__()

        # iNitializing the 7 user MIMO experiment
        # Args :
        #        H : Channel matrix
        #       se_max : maximum achievable spectral efficiency of 7 users

        # H.shape = (time, antennas, num of users)
        # each user has [MSE, DTU, Group] -> [Cureent Spectral efficiency potential, Fairness counter, Group idx]
        # so 7 users x 3 featues = 21 dimensional state, see difnationso of low and high

        self.H = H
        self.se_max = se_max
        self.num_ue = H.shape[2]
        self.curr_step = 0
        self.total_step = H.shape[0]
        ue_history = np.zeros((H.shape(2),))
        self.obs_sate = []
        self.usrgrp_cntr = []
        action_space_size = 127 #2^7 --> [0,126] INCLUSIVE
        self.action_space = gym.spaces.Discrete(action_space_size)

        #Minimum values for the state variables
        low = np.array([-np.inf , 0, 0]*7)
        #Maximum values for the state variables
        high = np.array([np.inf, np.inf, 6]*7)

        self.observation_space = gym.spaces.Box(low = np.array(low), high = np.array(high), dttype= np.float64)

        self.total_reward = None
        self.history = None

    
    def reset(self, seed=None, options = None):
        super().reset(seed=seed)

        """
        # Function defination to reset the environemt back to the initial state 
        Returns:
            numpy.ndarray: intital observation state
            dict: Information about the environment
        """

        self.curr_step = 0
        self.total_reward = 0
        self.history = {}
        self.jfi = 0
        self.sys_se = 0
        group_idx = user_group(np.squeeze(self.H[self.curr_step,:,:]))
        self.usrgrp_cntr.append(group_idx)
        self.ue_history = np.zeros((7,))
        initial_state = np.concatenate((np.reshape(self.se_max[self.current_step,:],(1,self.num_ue)),np.reshape(self.ue_history,(1,self.num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
        self.obs_sate.append(initial_state)
        info = self.getinfo()
        

        """
        Note : 
        The environment state contains three types of information:

        Channel quality → maximize throughput
        Scheduling history → fairness
        Group index → interference avoidance      

        So this is the multiobjective scheduling problem
        """

        """
        Users are clustered into groups based on channel similarity.
        Users are grouped according to channel state correlation
        Scheduling decisions consider user groups to avoid interference.

        Group 0 → similar channels
        Group 1 → similar channels
        Group 2 → similar channels
        """
        return initial_state, info
        
        
        
        