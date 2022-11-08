import math
import copy

import os
import random

import numpy as np
#from gym.utils import seeding
import time

#!pip install stable_baselines3
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from enviroments import *

# Pre-trainning Reptile Meta-Reinforcement Learning for Robust MTD

num_itr=10
num_task=20
alpha=1e-3
beta=1


num_config=4
max_rnd=100
#max_tau=max_rnd
initial_distribution=np.array([1,0,0,0])

M=np.array([[0,1,2,3],
            [2,0,1,2],
            [2,3,0,1],
            [3,2,1,0]])

r=10*np.random.randint(11, size=num_config)
#print(r)
q=0.1*np.random.randint(11, size=num_config)
#print(q)
env = Defender_Env(r,q)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, optimize_memory_usage=False, buffer_size=10000, gamma=1, 
             learning_rate= alpha, learning_starts=0, train_freq=(1, 'episode'))

model.save_replay_buffer("empty_replay_buffer")
meta_theta=model.policy.parameters_to_vector()

for i in range(num_itr):

    theta_lis=[]
    for _ in range(num_task):

        r=10*np.random.randint(11, size=num_config)
        print(r)
        q=0.1*np.random.randint(11, size=num_config)
        print(q)
        env = Defender_Env(r,q)
        model.set_env(env)
        model.policy.load_from_vector(meta_theta)
        for k in range(10):
            model.load_replay_buffer("empty_replay_buffer")
            #print(model.replay_buffer.size())
            model.learn(total_timesteps=100, log_interval=1, reset_num_timesteps=True)
        theta=model.policy.parameters_to_vector()
        #print(theta)
        theta_lis.append(theta)
    
    meta_theta=meta_theta-beta*(meta_theta-np.mean(theta_lis,axis=0))
    model.policy.load_from_vector(meta_theta)
    if (i+1)%2==0: model.save("meta_defender_TD3_{}".format(1000*(i+1)))

# remove to demonstrate saving and loading
del model

#adapt meta-policies

q=np.array([0.2,0.1,0.7,0.5])
r1=np.array([40,50,10,25])
r2=np.array([80,0,60,0])

env = adapt_defender_Env(r1,q)

model = TD3.load("meta_defender_TD3_10000.zip")

meta_theta=model.policy.parameters_to_vector()
print(meta_theta)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, optimize_memory_usage=False, buffer_size=10000, gamma=1, 
             learning_rate=1e-2, learning_starts=0, train_freq=(1, 'episode'))

model.set_env(env)
model.policy.load_from_vector(meta_theta)

model.learn(total_timesteps=100, log_interval=10, reset_num_timesteps=False) 
model.save("adapt1_defender_TD3_100")
model.learn(total_timesteps=100, log_interval=10, reset_num_timesteps=False) 
model.save("adapt1_defender_TD3_200")
model.learn(total_timesteps=100, log_interval=10, reset_num_timesteps=False) 
model.save("adapt1_defender_TD3_300")
model.learn(total_timesteps=100, log_interval=10, reset_num_timesteps=False) 
model.save("adapt1_defender_TD3_400")

del model

# test policies for 1000 steps


#model1 = TD3.load("best_defender_TD3_random_attack_10000.zip")
#model1 = TD3.load("defender_TD3_argmax_attack_10000.zip")
#model1 = TD3.load("meta_defender_TD3_10000.zip")
model1=TD3.load("adapt1_defender_TD3_100.zip")
#model1=TD3.load("random_initial.zip")



initial_distribution=np.array([1,0,0,0])
q=np.array([0.2,0.1,0.7,0.5])
r1=np.array([40,50,10,25])
r2=np.array([80,0,60,0])
r=r1

rewards=[]
for _ in range(1000):
    reward=0
    config=np.random.choice(num_config, p=initial_distribution)
    rnd=0
    #state=0
    #tau=0
    #tau_a=0
    #obs=np.array([config,tau,tau_a])
    #obs=np.array([config,state])
    obs=config

    for _ in range(100):
        rnd+=1
        
        #RL attack
        #att_action, _ = model2.predict(obs)
        #att_action=att_action/np.sum(att_action)
        
        #random attack
        #att_action=np.array([1/num_config]*num_config)
        
        #att_action=np.random.uniform(low=0.0, high=1.0, size=4)
        #att_action=att_action/np.sum(att_action)

        #att_config=np.random.choice(num_config, p=att_action)

        #p=0.5
        #def_action=np.array([(1-p)/(num_config-1)]*num_config)
        #def_action[config]=p
        #def_action=np.array([1/num_config]*num_config)
        def_action, _ = model1.predict(obs)
        def_action=def_action/np.sum(def_action)
        #print(def_action)

        #myopic argmax attack
        #att_will_lis=def_action*q*r1
        #if state==1: att_will_lis[config]*=1/q[config]
        #att_config=np.argmax(att_will_lis)
        att_config=np.argmax(def_action*q*r)

        old_config=config
        config = np.random.choice(num_config, p=def_action)
        #state=0
        #if config==old_config: tau+=1
        #else: 
            #tau=1
            #state=0
            #tau_a=0
        
        state=0
        if att_config==config and state==0: state=np.random.choice([0,1], p=[1-q[config], q[config]])
        #if att_config==config and config==old_config: state=1
        #if att_config==config and config!=old_config: state=np.random.choice([0,1], p=[1-q[config], q[config]])
        #if state==1: tau_a+=1

        #reward+=state*r2[config]
        #else: reward+=M[old_config, config]
        reward+=state*r[config]+M[old_config, config]


        #obs=np.array([config,tau, tau_a])
        #obs=np.array([config,state])
        obs=config

    #print(reward)
    rewards.append(reward)

avg_reward=np.average(rewards)
print(avg_reward)

