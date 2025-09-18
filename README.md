# Thesis Code

This repository contains the implementation used in my MSc thesis on **MPC-based Reinforcement Learning with Control Barrier Functions (CBFs)**.

## Project Summary

Optimal control strategies are often combined with safety certificates to ensure both performance and safety in safety-critical systems. A common approach is combining **Model Predictive Control (MPC)** with **Control Barrier Function (CBF)** constraints, but tuning MPC parameters and selecting an appropriate class $\mathcal{K}$ function in the CBF is highly problem-dependent.

This project implements a **safe model-based reinforcement learning (RL) framework** where a parameterized MPC incorporates a CBF with a learnable $\mathcal{K}$ function. The framework introduces three variations:

- **LOPTD-CBF (Learnable Optimal Decay CBF):** extends the Optimal Decay CBF by letting RL tune decay parameters, enhancing performance while preserving feasibility and safety guarantees.  
- **SNN-CBF (Sigmoid Neural Network CBF):** parameterizes the decay term of a discrete exponential CBF with a small neural network, enabling richer, state-dependent safety conditions.  
- **SRNN-CBF (Sigmoid Recurrent Neural Network CBF):** extends SNN-CBF with a recurrent architecture to handle **time-varying constraints**, such as moving obstacles, by anticipating their motion.  

Numerical experiments on a **discrete double-integrator system** with static and dynamic obstacles demonstrate that the methods improve performance while ensuring safety, each offering distinct trade-offs in interpretability, complexity, and feasibility.

---

## Repository Structure

- **`main.py`**  
  Entry point for running training and evaluation.  
  - Configures experiment parameters (MPC horizon, RL learning rate, penalties, noise schedule, etc.).  
  - Initializes the RNN-based CBF inside the MPC.  
  - Runs simulations before and after training.  
  - Executes the RL training loop and logs results.

- **`Classes.py`**  
  Core classes for the environment and control framework:  
  - `env`: Linear system dynamics wrapped as a Gymnasium environment.  
  - `MPC`: Formulation of the nonlinear program (CasADi + IPOPT) with CBF constraints and slack handling.  
  - `Obstacles` & `ObstacleMotion`: Define static and moving obstacle dynamics (e.g., step-bounce, orbit, sinusoid).  
  - `RNN`: Elman-style recurrent neural network predicting CBF parameters (`α` values).  
  - `RLclass`: Reinforcement learning loop for updating MPC/CBF parameters.

- **`Functions.py`**  
  Helper functions for running experiments:  
  - MPC solvers, stage cost computation, and rollout functions.  
  - Visualization utilities (animated trajectories with obstacles, predicted horizons, constraint satisfaction plots).  
  - Experiment logging and result saving.

- **`config.py`** *(not shown above, but imported)*  
  Stores simulation constants: sampling time, system dimensions, state/input constraints, random seed, etc.

---
