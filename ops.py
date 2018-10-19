import torch
import torch.nn as nn
import gym
from itertools import count
import numpy as np
import random
import torch.nn.functional as F
from models import Creature
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
def measure_fitness(creature,env,device,discrete_actions,min_reward,render = False,max_steps = 1000):
    observation = env.reset()
    
    #creature fitness is cumulative reward in simulation
    total_reward = 0
    i = 0
    while True:
        if (i >= max_steps and max_steps > 0) or total_reward < min_reward:
            break
            
        if render:
            env.render()
            
        #convert observation into tensor
        obs = torch.from_numpy(observation).type('torch.FloatTensor').to(device)
       
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
        i+=1
    return total_reward

#measure fitness of entire population and return scores
def measure_population_fitness(population,env,device,discrete_actions,min_reward,max_steps = 1000):
    scores = []

    for idx,p in enumerate(population):
        fitness = measure_fitness(p,env,device,discrete_actions,min_reward,max_steps = max_steps)
        scores.append(fitness)
    return np.array(scores)

def write(memory,w_,e,a,device):
    w = w_.squeeze(0).unsqueeze(-1)*torch.ones([1,memory.size(-1)]).type("torch.FloatTensor").to(device).unsqueeze(0)
    w = F.cosine_similarity(w.transpose(1,2),memory.transpose(1,2))
    #print(w)
    w = torch.softmax(w,-1)
    erase = (w.unsqueeze(1)*e.unsqueeze(2))
    add = (w.unsqueeze(1)*a.unsqueeze(2))
    return ((memory.transpose(1,2)*(1-erase))+add).transpose(1,2)

def read(memory,w_,device):
    w = w_.squeeze(0).unsqueeze(-1)*torch.ones([1,memory.size(-1)]).type("torch.FloatTensor").to(device).unsqueeze(0)
    w = F.cosine_similarity(w.transpose(1,2),memory.transpose(1,2))
    w = torch.softmax(w,-1)
    batch = []
   
    for b in w:
        batch.append(torch.matmul(memory.transpose(1,2),b))
    return torch.stack(batch)#.squeeze().unsqueeze(0)

def mate(env,creature_out_size,a,device,m,d,mutation_rate_m,mutation_rate_d,use_gen,mutation_scale=0.07):
    child = Creature(env.observation_space.shape[0],creature_out_size)
    #mom = mutate(mom,mutation_rate_m,mutation_scale)
    #dad = mutate(dad,mutation_rate_d,mutation_scale)
    
    #get parents as vectors
    mom = get_params(m)
    dad = get_params(d)
    
    a = np.array([a])
    a = torch.from_numpy(a).type("torch.FloatTensor").to(device)

    generated = use_gen(mom,dad,a).squeeze(0)
    child = set_params(child,generated)
    
    #mutate child
    mutation_rate = np.min([mutation_rate_m,mutation_rate_d])
    child = mutate(child,device,mutation_rate,mutation_scale)
    
    return child

def mutate(creature,device,mutation_rate=0.2,scale = 0.07,start_layer = 0):
    if mutation_rate != 0:
        new = creature.__class__(creature.input_size,creature.output_size).to(device)
        new.load_state_dict(creature.state_dict()) 
        for idx,p in enumerate(new.parameters()):
            if idx < start_layer:
                continue
            mutation = np.random.normal(scale = scale,size = p.data.shape)
            mutation *= np.random.choice([1, 0], p.data.shape,p=[mutation_rate,1-mutation_rate])
            mutation = torch.from_numpy(mutation).type('torch.FloatTensor').to(device)
            p.data += mutation
        return new
    else:
        return creature

def gen_children(population,device,use_gen,batch_size, a = 0.1):
    mom = []
    dad = []
    child = []
    for b in range(batch_size):
        m = get_params(random.choice(population))
        d = get_params(random.choice(population))

        a = np.array([a])
        a = torch.from_numpy(a).type("torch.FloatTensor").to(device)


        c = use_gen(m,d,a).squeeze(0)

        mom.append(m)
        dad.append(d)
        child.append(c)

    mom = torch.stack(mom).to(device)
    dad = torch.stack(dad).to(device)
    child = torch.stack(child).to(device)
    return child    
        