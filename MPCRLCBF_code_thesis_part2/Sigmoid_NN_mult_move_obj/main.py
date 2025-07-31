
import numpy as np
import casadi as cs
import os
import copy
from config import SAMPLING_TIME, SEED, NUM_STATES, NUM_INPUTS, CONSTRAINTS_X, CONSTRAINTS_U
from Classes import env
from Classes import RLclass
from Classes import NN
from Functions import run_simulation, run_simulation_randomMPC, generate_experiment_notes





######### Main #########
#parameters for running the experiments

dt = SAMPLING_TIME
seed = SEED
noise_scalingfactor = 20
noise_variance = 5
alpha = 5e-1
gamma = 0.95
episode_duration= 150
num_episodes = 3000

# MPC HORIZON
horizon = 5

replay_buffer= 10*episode_duration #buffer is 5 episodes long
episode_updatefreq = 10# updates every 3 episodes

patience_threshold = 10000
lr_decay_factor = 0.1

decay_at_end = 0.01
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

# # Obstacles used in the enviroment
# positions = [(-1.75, -1.5), (-4.0, -3.0)]
# radii     = [0.5, 0.5]

# modes       = ["step_bounce", "step_bounce"]                   # one obstacle, bouncing
# mode_params = [{"bounds":(-4.0, 1.0), "speed":1.5, "dir": 1},  {"bounds":(-4.0, 1.0), "speed":1.5, "dir": -1}]

positions = [(-2.0, -1.5), 
             (-3.0, -3.0)]


radii     = [0.75, 0.75]  

modes       = ["step_bounce", "step_bounce"]
mode_params = [{'bounds': (-4.0, 0.0), 'speed': 2.3, 'dir': 1}, 
               {'bounds': (-4.0, 1.0), 'speed': 2.0, 'dir': -1}]

# positions = [
#     ( -1.0, -1.0),  # orbit #1 around origin
#     (-2.0, 2.0),  # orbit #2 around (–1,1)
#     (2.0, -2.0),
#     (1.0, 1.0),
# ]

# radii = [0.7, 1.5, 1.5, 0.7]  # radii of the obstacles

# modes = [
#     "orbit",
#     "orbit",
#     "orbit",
#     "orbit",
# ]

# mode_params = [
#     # orbit #1: angular speed 0.8 rad/s around origin
#     {"omega": 0.5, "center": (0.0, 0.0)},
#     # orbit #2: angular speed 1.2 rad/s around (-1, 1)
#     {"omega": 0.5, "center": (0.0, 0.0)},
#     {"omega": 0.5, "center": (0.0, 0.0)},
#     {"omega": 0.5, "center": (0.0, 0.0)},
# ]

# inputs --> states and the different h(x)
# last layer output --> number of obstacles aka for each h(x) have an alpha value

layers_list = [NUM_STATES+len(positions), 14, 14, 14, len(positions)]
print("layers_list: ", layers_list)
#initializing the NN
nn_model = NN(layers_list, positions, radii)

list, _, _ = nn_model.initialize_parameters()

params_innit["nn_params"] = list

#seems params_innit gets overwritten
params_original = params_innit.copy() 

experiment_folder_name = "NN_mult_move_obj_experiment_44"

#WATCH OUT --> since mode_params was a dict what was happening is that the values of it kept getting changed

run_simulation_randomMPC(params_innit, env, experiment_folder_name, episode_duration, layers_list, noise_scalingfactor, noise_variance, 
horizon, positions, radii, modes, copy.deepcopy(mode_params))

stage_cost_sum_before = run_simulation(params_innit, env, experiment_folder_name, episode_duration, layers_list, False, horizon, positions, radii, modes, copy.deepcopy(mode_params))

rl = RLclass(params_innit, seed, alpha, gamma, decay_rate, layers_list, noise_scalingfactor, 
             noise_variance, patience_threshold, lr_decay_factor, horizon, positions, radii, modes, copy.deepcopy(mode_params))

params = rl.rl_trainingloop(episode_duration = episode_duration, num_episodes = num_episodes, replay_buffer=replay_buffer,
                            episode_updatefreq = episode_updatefreq, experiment_folder = experiment_folder_name)

stage_cost_sum_after = run_simulation(params, env, experiment_folder_name, episode_duration, layers_list, True, horizon, positions, radii, modes, copy.deepcopy(mode_params))


generate_experiment_notes(experiment_folder_name, params, params_original, episode_duration, num_episodes, seed, alpha, dt, 
                          gamma, decay_rate, decay_at_end, noise_scalingfactor, noise_variance, stage_cost_sum_before, 
                          stage_cost_sum_after, layers_list, replay_buffer, episode_updatefreq, patience_threshold, lr_decay_factor, horizon,
                          modes, copy.deepcopy(mode_params), positions, radii)


suffix = f"_stagecost_{stage_cost_sum_after:.2f}"

new_folder_name = experiment_folder_name + suffix

# Rename the existing folder name
os.rename(experiment_folder_name, new_folder_name)




# PROBLEMS:

# I noticed that i punished cost for using slacks differently in MPC cost func and RL cost func
# MPC cost func: np.sum(2e6 *S)
# RL cost func: 2e6 * np.sum(S)

# SOLUTION: I changed the RL cost func to match the MPC one







