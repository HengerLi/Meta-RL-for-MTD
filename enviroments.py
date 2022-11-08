import gym
from gym import spaces

import numpy as np
import time
import os

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

num_config=4
max_rnd=100
#max_tau=max_rnd
initial_distribution=np.array([1,0,0,0])

M=np.array([[0,1,2,3],
            [2,0,1,2],
            [2,3,0,1],
            [3,2,1,0]])

q=np.array([0.2,0.1,0.7,0.5])
r1=np.array([40,50,10,25])
r2=np.array([80,0,60,0])


__all__=['Attacker_Env', 'Defender_Env','adapt_defender_Env',]

class Attacker_Env(gym.Env):

    def __init__(self):
        
        self.rnd=0

        high = 1
        low = 0
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(num_config,),
            dtype=np.float32
        )

        #self.observation_space=spaces.MultiDiscrete([num_config, max_tau, max_tau])
        self.observation_space=spaces.MultiDiscrete([num_config, 2])
        


    def step(self, action):
        
        
        self.rnd+=1
        #print(action)
        
        #att_action=action+1
        att_action=action/np.sum(action)
        #print(action)
        #att_config=np.argmax(action*q*g)
        
        if self.training_length==0: def_action=np.array([1/num_config]*num_config)
        else: 
            def_action, _ = self.model.predict(self.obs)
            def_action=att_action/np.sum(att_action)

        att_config=np.random.choice(num_config, p=att_action)
        
        old_config=self.config
        self.config = np.random.choice(num_config, p=def_action)
        if self.config==old_config: self.tau+=1
        else: 
            self.tau=1
            self.state=0
            self.tau_a=0
        
        if att_config==self.config and self.state==0: self.state=np.random.choice([0,1], p=[1-q[self.config], q[self.config]])
        if self.state==1: self.tau_a+=1

        reward=self.state*r1[self.config]
        #if math.isinf(reward) or math.isnan(reward):
            #print("abnormal reward")
            #reward = 0
        
        done=False
        if self.rnd==max_rnd: done=True

        #self.obs=np.array([self.config,self.tau, self.tau_a])
        self.obs=np.array([self.config,self.state])
        
        return self.obs, reward, done, {}

    def reset(self):

        self.rnd=0
        self.config=np.random.choice(num_config, p=initial_distribution)
        self.state=0
        self.tau=0
        self.tau_a=0
        
        self.training_length=110000
        while os.path.exists("ALTdefender_TD3_"+str(self.training_length)+".zip")==False:
            self.training_length-=10000
            #print(self.training_length)
            if self.training_length==0: break
            if os.path.exists("ALTdefender_TD3_"+str(self.training_length)+".zip"):
                self.model = TD3.load("ALTdefender_TD3_"+str(self.training_length)+".zip")
        
        #self.obs=np.array([self.config,self.tau, self.tau_a])
        self.obs=np.array([self.config,self.state])
        
        return self.obs


class Defender_Env(gym.Env):

    def __init__(self,r,q):
        
        self.rnd=0
        self.r=r
        self.q=q

        high = 1
        low = 0
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(num_config,),
            dtype=np.float32
        )

        #self.observation_space=spaces.MultiDiscrete([num_config, max_tau, max_tau])
        #self.observation_space=spaces.MultiDiscrete([num_config, 2])
        self.observation_space=spaces.Discrete(num_config)
        


    def step(self, action):
        
        
        self.rnd+=1
        #print(action)
        
        def_action=action/np.sum(action)
        #print(action)
        
        #att_config=np.argmax(def_action*q*r)
        
        #if self.training_length==0:
            #print('random attack!') 
            #att_action=np.array([1/num_config]*num_config)
        #else: 
            #att_action, _ = self.model.predict(self.obs)
            #att_action=att_action/np.sum(att_action)
        
        if self.mod==1:  
            att_action=np.array([1/num_config]*num_config)
            att_config=np.random.choice(num_config, p=att_action)
        
        else: att_config=np.argmax(def_action*self.q*self.r)
        
        
        old_config=self.config
        self.config = np.random.choice(num_config, p=def_action)
        #if self.config==old_config: self.tau+=1
        #else: 
            #self.tau=1
            #self.state=0
            #self.tau_a=0
        self.state=0
        if att_config==self.config and self.state==0: self.state=np.random.choice([0,1], p=[1-self.q[self.config], self.q[self.config]])
        #if self.state==1: self.tau_a+=1

        #if self.config==old_config: reward=-self.state*self.r[self.config]
        #else: reward=-M[old_config, self.config]
        reward=-self.state*self.r[self.config]-M[old_config, self.config]
        
        
        #print(self.state)

        
        done=False
        if self.rnd==max_rnd: done=True

        #self.obs=np.array([self.config,self.tau, self.tau_a])
        #self.obs=np.array([self.config,self.state])
        self.obs=self.config
        
        return self.obs, reward, done, {}

    def reset(self):

        self.rnd=0
        self.config=np.random.choice(num_config, p=initial_distribution)
        #self.state=0
        #self.tau=0
        #self.tau_a=0
        
        #self.training_length=110000
        #while os.path.exists("ALTattacker_TD3_"+str(self.training_length)+".zip")==False:
            #self.training_length-=10000
            #print(self.training_length)
            #if self.training_length==0: break
            #if os.path.exists("ALTattacker_TD3_"+str(self.training_length)+".zip"):
                #self.model = TD3.load("ALTattacker_TD3_"+str(self.training_length)+".zip")
        
        self.mod=np.random.choice([0,1])
        #if self.mod<=2: self.r=r1
        #else: self.r=r2
        

        #self.obs=np.array([self.config,self.tau, self.tau_a])
        #self.obs=np.array([self.config,self.state])
        self.obs=self.config

        return self.obs


class adapt_defender_Env(gym.Env):

    def __init__(self,r,q):
        
        self.rnd=0
        self.r=r
        self.q=q

        high = 1
        low = 0
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(num_config,),
            dtype=np.float32
        )

        #self.observation_space=spaces.MultiDiscrete([num_config, max_tau, max_tau])
        #self.observation_space=spaces.MultiDiscrete([num_config, 2])
        self.observation_space=spaces.Discrete(num_config)
        


    def step(self, action):
        
        
        self.rnd+=1
        #print(action)
        
        def_action=action/np.sum(action)
        #print(action)
        
        #att_config=np.argmax(def_action*q*r)
        
        #if self.training_length==0:
            #print('random attack!') 
            #att_action=np.array([1/num_config]*num_config)
        #else: 
            #att_action, _ = self.model.predict(self.obs)
            #att_action=att_action/np.sum(att_action)
        
        if self.mod==1:  
            att_action=np.array([1/num_config]*num_config)
            att_config=np.random.choice(num_config, p=att_action)
        
        else: att_config=np.argmax(def_action*self.q*self.r)
        
        
        old_config=self.config
        self.config = np.random.choice(num_config, p=def_action)
        #if self.config==old_config: self.tau+=1
        #else: 
            #self.tau=1
            #self.state=0
            #self.tau_a=0
        self.state=0
        if att_config==self.config and self.state==0: self.state=np.random.choice([0,1], p=[1-self.q[self.config], self.q[self.config]])
        #if self.state==1: self.tau_a+=1

        #if self.config==old_config: reward=-self.state*self.r[self.config]
        #else: reward=-M[old_config, self.config]
        reward=-self.state*self.r[self.config]-M[old_config, self.config]
        
        
        #print(self.state)

        
        done=False
        if self.rnd==max_rnd: done=True

        #self.obs=np.array([self.config,self.tau, self.tau_a])
        #self.obs=np.array([self.config,self.state])
        self.obs=self.config
        
        return self.obs, reward, done, {}

    def reset(self):

        self.rnd=0
        self.config=np.random.choice(num_config, p=initial_distribution)
        #self.state=0
        #self.tau=0
        #self.tau_a=0
        
        #self.training_length=110000
        #while os.path.exists("ALTattacker_TD3_"+str(self.training_length)+".zip")==False:
            #self.training_length-=10000
            #print(self.training_length)
            #if self.training_length==0: break
            #if os.path.exists("ALTattacker_TD3_"+str(self.training_length)+".zip"):
                #self.model = TD3.load("ALTattacker_TD3_"+str(self.training_length)+".zip")
        
        #self.mod=np.random.choice([0,1])
        self.mod=0
        #if self.mod<=2: self.r=r1
        #else: self.r=r2
        

        #self.obs=np.array([self.config,self.tau, self.tau_a])
        #self.obs=np.array([self.config,self.state])
        self.obs=self.config

        return self.obs