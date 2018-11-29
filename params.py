population_size = 10
max_population = population_size
max_population_batch = 2
out_size = population_size
n_generations = 300 
all_a = 0.0001
population_exponent = -1

envs = ['CartPole-v1','Acrobot-v1','MountainCar-v0','Pendulum-v0','BipedalWalker-v2','LunarLander-v2']
env_to_use = -2

discrete_actions = False

mutation_scale = 0#.1
creature_mutation_rate = 0.7
mutation_rate = 0.1
