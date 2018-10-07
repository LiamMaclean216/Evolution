import torch
import torch.nn as nn
import gym
from itertools import count
import numpy as np
import random
import torch.nn.functional as F

#turn vector into model parameters
def set_params(model,data):
    idx = 0
    for p in model.parameters():
        view = data[idx:idx+p.numel()].view(p.shape)
        p.data = view
        idx+=p.numel()
    return model

#get model parameters as vector
def get_params(model):
    params = []
    for p in model.parameters():
        view = p.view(p.numel())
        params.append(view)
    params = torch.cat(params, dim=0)
    return params

#get probability of creature getting picked based on fitness
def get_pick_probabilities(p_fitness):
    if(np.min(p_fitness) == np.max(p_fitness)):
        return np.ones(p_fitness.shape)/p_fitness.size
    
    pick_probabilities = (p_fitness-np.min(p_fitness))
    pick_probabilities = pick_probabilities/np.sum(pick_probabilities)
    return pick_probabilities


#measure creature fitness
def measure_fitness(creature,env,device,discrete_actions,render = False,max_steps = 1000):
    observation = env.reset()
    #creature fitness is cumulative reward in simulation
    total_reward = 0
    
    for i in range(max_steps):
        
        if render:
            env.render()
            
        #convert observation into tensor
        obs = torch.from_numpy(observation).to(device).type('torch.cuda.FloatTensor')
       
        #get action
        if discrete_actions:
            action = creature(obs)
            sample = (obs,action)
            action = action.max(-1)[1].item()
        else:
            action = creature(obs)
            sample = (obs,action)
            action = action.detach().cpu().numpy()
            
        observation, reward, done, _ = env.step(action)
        
        total_reward += reward
        
        if done:
            break
    
    return total_reward

#measure fitness of entire population and return scores
def measure_population_fitness(population,env,device,discrete_actions,max_steps = 1000):
    scores = []

    for idx,p in enumerate(population):
        fitness = measure_fitness(p,env,device,discrete_actions,max_steps = max_steps)
        scores.append(fitness)
    return np.array(scores)