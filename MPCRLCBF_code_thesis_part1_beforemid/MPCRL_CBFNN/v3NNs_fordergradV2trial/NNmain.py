
import numpy as np
import casadi as cs
import os
from NNclasses import env
from NNclasses import RLclass
from NNclasses import NN
from NNfunctions import run_simulation, run_simulation_randomMPC, generate_experiment_notes





######### Main #########
#parameters for running the experiments

dt = 0.2
seed = 69
noise_scalingfactor = 4
noise_variance = 5
alpha = 5e-1
gamma = 0.95
episode_duration= 3000
num_episodes = 3000

layers_list = [5, 8, 8, 8, 1]

replay_buffer= 10*episode_duration #buffer is 5 episodes long
episode_updatefreq = 10# updates every 3 episodes

patience_threshold = 500000
lr_decay_factor = 0.1

decay_at_end = 0.001
decay_rate = 1 - np.power(decay_at_end, 1/(num_episodes/episode_updatefreq))
print(f"decay_rate: {decay_rate}")


params_innit = {
    # State matrices
    "A": cs.DM([
            [1, 0, dt, 0], 
            [0, 1, 0, dt], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ]) ,
    "B": cs.DM([
            [0.5 * dt**2, 0], 
            [0, 0.5 * dt**2], 
            [dt, 0], 
            [0, dt]
        ]),
    # Learned parameters
    "b":  cs.DM([0, 0, 0, 0]),
    "V0": cs.DM(0.0),
    "P" : 100*np.identity(4),
    "Q" : 10*np.identity(4), 
    "R" : np.identity(2)
}

#initializing the NN
nn_model = NN(layers_list)

list, _, _ = nn_model.initialize_parameters()

params_innit["nn_params"] = list

#seems params_innit gets overwritten
params_original = params_innit.copy() 

experiment_folder_name = "NNforderADAM_22"



run_simulation_randomMPC(params_innit, env, experiment_folder_name, episode_duration, layers_list, noise_scalingfactor, noise_variance)

stage_cost_sum_before = run_simulation(params_innit, env, experiment_folder_name, episode_duration, layers_list, False)

rl = RLclass(params_innit, seed, alpha, dt, gamma, decay_rate, layers_list, noise_scalingfactor, noise_variance, patience_threshold, lr_decay_factor)
params = rl.rl_trainingloop(episode_duration = episode_duration, num_episodes = num_episodes, replay_buffer=replay_buffer, episode_updatefreq = episode_updatefreq, experiment_folder = experiment_folder_name)

stage_cost_sum_after = run_simulation(params, env, experiment_folder_name, episode_duration, layers_list, True)


generate_experiment_notes(experiment_folder_name, params, params_original, episode_duration, num_episodes, seed, alpha, dt, 
                          gamma, decay_rate, decay_at_end, noise_scalingfactor, noise_variance, stage_cost_sum_before, 
                          stage_cost_sum_after, layers_list, replay_buffer, episode_updatefreq, patience_threshold, lr_decay_factor)


suffix = f"_stagecost_{stage_cost_sum_after:.2f}"

new_folder_name = experiment_folder_name + suffix

# Rename the existing folder name
os.rename(experiment_folder_name, new_folder_name)







