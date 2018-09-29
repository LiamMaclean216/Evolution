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

def run_params(model,model_data,input):
    idx = 0
    out = input
    for i,p in enumerate(model.parameters()):
        view = model_data[idx:idx+p.numel()].view(p.size())
    
        
        idx+=p.numel()
        
        
        if i % 2 == 0:
            out = torch.matmul(out, view.transpose(0,1))
        else :
            out += view
            
    return out

#get model parameters as vector
def get_params(model):
    params = []
    for p in model.parameters():
        view = p.view(p.numel())
        params.append(view)
    params = torch.cat(params, dim=0)
    return params

def get_pick_probabilities(p_fitness):
    normed = p_fitness- np.mean(p_fitness)
    normed -= np.min(normed)
    normed = np.power(normed, 0.5)
    pick_probabilities = normed/(np.sum(normed))
    return pick_probabilities



#measure creature fitness
def measure_fitness(creature,env,device,discrete_actions,render = False,max_steps = 1000,n_behavior_samples = 10):
    observation = env.reset()
    #creature fitness is cumulative reward in simulation
    total_reward = 0
    
    #sample behavior from episode for autoencoder training
    behavior_samples = []
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
            
        #add current sample to all behavior samples
        
        behavior_samples.append(sample)
        
        observation, reward, done, _ = env.step(action)
        
        total_reward += reward
        
        if done:
            break
    
    #reshape behavior samples into tensors
    behavior_samples = random.sample(behavior_samples,min(n_behavior_samples,len(behavior_samples)))
    act,obs = zip(*behavior_samples)
    return total_reward, torch.stack(act,0), torch.stack(obs,0)

#measure fitness of entire population and return scores
def measure_population_fitness(population,env,device,discrete_actions,max_steps = 1000,n_behavior_samples = 10):
    scores = []
    actions = []
    observations = []
    for idx,p in enumerate(population):
       #print("measuring fitness : {}".format(idx))
        fitness, act,obs = measure_fitness(p,env,device,discrete_actions,max_steps = max_steps,n_behavior_samples=n_behavior_samples)
        scores.append(fitness)
        
        actions.append(act)
        observations.append(obs)

    return np.array(scores),(actions,observations)