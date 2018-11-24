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

    for p in population:
        fitness = measure_fitness(p,env,device,discrete_actions,min_reward,max_steps = max_steps)
        scores.append(fitness)
    return np.array(scores)

def mate(env,creature_out_size,a,device,m,d,mutation_rate_m,mutation_rate_d,use_gen,mutation_scale=0.07):
    child = Creature(env.observation_space.shape[0],creature_out_size)
    
    #get parents as vectors
    mom = get_params(m)
    dad = get_params(d)
    
    #a = np.array([a])
    #a = torch.from_numpy(a).type("torch.FloatTensor").to(device)
    a = torch.from_numpy(np.array([a])).type('torch.FloatTensor').to(device)
    
    generated,confidence = use_gen(mom.unsqueeze(0),dad.unsqueeze(0),a)
    generated = generated.squeeze(0)
    confidence = confidence.squeeze(0)
    mutation_rate = np.min([mutation_rate_m,mutation_rate_d])
    generated = mutate(generated,device,confidence,mutation_rate,mutation_scale)
    child = set_params(child,generated)
    

    return child

def mutate(creature,device,confidence,mutation_rate=0.2,scale = 0.07):
    out = creature
    if mutation_rate != 0 and scale != 0:
        #print("#")
        #print(confidence[0:10])
        mutation = np.random.normal(scale = scale,size = creature.shape)
        mutation *= ((1-np.abs(confidence.cpu().detach().numpy()))) + 0.1
        #print(mutation[0:10])
        mutation *= np.random.choice([1, 0], creature.shape,p=[mutation_rate,1-mutation_rate])
       
        mutation = torch.from_numpy(mutation).type('torch.FloatTensor').to(device)
        out += mutation
        return out
    else:
        return creature

def gen_children(population,device,use_gen,batch_size, a = 0.1):
    child = []
    m_batch = []
    d_batch = []
    for b in range(batch_size):
        #m = get_params(random.choice(population))
        #d = get_params(random.choice(population))
        m_batch.append(get_params(random.choice(population)))
        d_batch.append(get_params(random.choice(population)))
        #a = np.array([a])
        #a = torch.from_numpy(a).type("torch.FloatTensor").to(device)
    m_batch = torch.stack(m_batch, dim=0).to(device)
    d_batch = torch.stack(d_batch, dim=0).to(device)
    
    c,confidence = use_gen(m_batch,d_batch,a)
    c = c.squeeze(0)
    #print(gen_a)
    child.append(c)
        
    #all_a = torch.stack(all_a).to(device)
    child = torch.stack(child).to(device)
    return c ,confidence   
        
def gen_children_no_batch(population,device,gen,batch_size, a = 0.1):
    child = []
    gen_a_all = [] 
    np.random.shuffle(population)
    for b in range(batch_size):
        
        #m = get_params(random.choice(population)).unsqueeze(0)
        #d = get_params(random.choice(population)).unsqueeze(0)
        m = get_params(population[b]).unsqueeze(0)
        if b < len(population)-1:
            d = get_params(population[b+1]).unsqueeze(0)
        else:
            d = get_params(population[0]).unsqueeze(0)
        c,_,gen_a = gen(m,d,a[b].unsqueeze(-1))
        c = c.squeeze(0)
        child.append(c)
        gen_a_all.append(gen_a)
    child = torch.stack(child).to(device)
    gen_a = torch.stack(gen_a_all).to(device)
    return child, gen_a
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    