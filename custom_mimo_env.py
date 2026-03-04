# ======================== IMPORTS ==================================
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

# ==================================================================


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
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    """
    [=] Flow of the STEP of agent in the enviroment:
    Steps:
        1) Agents picks the scheduling action
        2) User selected
        3) Channel matrix extracted (No of antennas x selected Users)
        4) MIMO transmissiojn simulated
        5) Spectral efficiency calculated
        6) Reward computed
        7) Next state generated
    """
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def step(self, action):

        """
        Defination of agent step in environment

        Args: 
            action taken by the agent
        Returns:
            numpy.ndarray => latest observation state (Next)
            float => reward of the action
            bool => episode termination 
            dict => Information about the environment
        """
        
        # 1) Agent is picking the action
        # 2) user selection
        ue_select , idx = sel_ue(action)

        #3) Channel matrix extracted (No of antennas x selected Users)
        #4) MIMO transmissiojn simulated

        #Modulation selection: Sane for all (log2(4)) 
        mod_select = np.ones((idx,)) * 4
        #antennas x selected users will be H 
        ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(self.H[self.current_step,:,ue_select],(64,-1)),idx,mod_select)
        
        #5)6) Ccaluations
        #reward, history, jfi ,se calculatikons
        reward, self.ue_history, jfi, sys_se = self.calculate_reward(ur_se_total, ur_min_snr, ur_se, ue_select, idx, self.usrgrp_cntr[self.current_step], self.se_max[self.current_step])
        self.jfi = jfi 
        self.sys_se = sys_se
        self.total_reward = self.total_reward + reward
        
        self.curr_step += 1
        done_pm = self.total_step - 1
        done = self.curr_step >= self.total_step
        truncated = False

        #getting group index
        group_idx = usr_group(np.squeeze(self.H[(self.curr_step),:,:]))
        self.usrgrp_cntr.append(group_idx)

        #7)8) next state difinaton
        next_state = np.concatenate((np.reshape(self.se_max[(self.current_step),:],(1,self.num_ue)),np.reshape(self.ue_history,(1,self.num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
        self.obs_state.append(next_state)       

        info = self.getinfo()
        history = self.update_history(info)   


        return next_state, reward, done, truncated, info

    def get_reward(self, action):

        """
        This is the reward calculation defination

        Args:
            action 
        Returns:
            float reward
        """
        
        # Saves the current state
        current_step = self.current_step
        ue_history = self.ue_history.copy()

        #Reward calculations
        ue_select, idx = sel_ue(action)
        mod_select = np.ones((idx,))* 4
        ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(self.H[current_step, :, ue_select], (64, -1)), idx, mod_select)
        reward, _, _, _ = self.calculate_reward(ur_se_total, ur_min_snr, ur_se, ue_select, idx, self.usrgrp_cntr[current_step], self.se_max[current_step], se_noise=True)
        
        # Restores the state
        self.current_step = current_step
        self.ue_history = ue_history

        return reward
    
    
        