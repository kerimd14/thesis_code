import gymnasium as gym 
import numpy as np
import os # to communicate with the operating system
import copy
from gymnasium.spaces import Box
import casadi as cs
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['axes.formatter.use_mathtext'] = False
import matplotlib.pyplot as plt
from control import dlqr
from collections import deque
import pandas as pd
import matplotlib.animation as animation

from config import SAMPLING_TIME, NUM_INPUTS, NUM_STATES, CONSTRAINTS_X, SEED, CONSTRAINTS_U



class Obstacles:

    def __init__(self, positions, radii):

        """
        Initialize obstacles.

        Args:
            positions: List of (x, y) obstacle centers.
            radii:     List of obstacle radii.
        """
        if len(positions) != len(radii):
            raise ValueError(f"Expected same number of positions and radii, "
                            f"but got {len(positions)} positions and {len(radii)} radii.")
        self.positions = positions
        self.radii = radii
        self.obstacle_num = len(positions)
        self.dt = SAMPLING_TIME

    def h_obsfunc(self, x, xpred_list, ypred_list):
        """
        [NOT USED IN CODE]
        
        Numerically evaluate CBF h_i(x) = (x - x_pred[k])^2 + (y - y_pred[k])^2 - r^2
        for each obstacle over the dynamic constraints.
        
        Args:
            x:       Current state vector [x, y, …].
            xpred:   Flattened list of predicted obstacle x-positions.
            ypred:   Flattened list of predicted obstacle y-positions.

        Returns:
            A list of h_i(x) values (one per obstacle). 
        """
        #vx and vy are supposed represent the change in postion of obstacles, that is how i will show velocity
        h_list = []
        
        for (r, x_pred, y_pred)in zip(self.radii, xpred_list, ypred_list):
            
            h_list.append( (x[0] - x_pred)**2 + (x[1] - y_pred)**2 - r**2 )
            
        return h_list

    def make_h_functions(self):
        """
        Build symbolic CasADi Functions for each obstacle h_i(x) i for 1,...,obstacle_num (not pythony index)

        Returns:
            A list of CasADi Functions for each obstacle over the dynamic obstacle constraints.
            Function Inputs: (x: MX[4x1], xpred_list: MX[num], ypred_list: MX[num])
            Funciton outputs: h_i : MX[1x1].
        """
        funcs = []
        
        #states
        x = cs.MX.sym("x", 4)
        
        #vx and vy are supposed represent the change in postion of obstacles, that is how i will show velocity
        xpred_list = cs.MX.sym("vx", self.obstacle_num)  # velocity in x direction
        ypred_list= cs.MX.sym("vy", self.obstacle_num) # velocity in y direction

         # Build one function per obstacle 
        for  idx, (r) in enumerate(self.radii):
            xpred_i    = xpred_list[idx]
            ypred_i    = ypred_list[idx]
            
            hi_expr = (x[0] - xpred_i)**2 + (x[1] - ypred_i)**2 - r**2

            hi_fun = cs.Function(f"h{idx+1}", [x, xpred_list, ypred_list], [hi_expr])
            funcs.append(hi_fun)

        return funcs
    
    

class ObstacleMotion:
    def __init__(self,
                 positions: list,
                 modes: list[str],
                 mode_params: list[dict]):
        """
        Responsible for taking care of the motion of the obstacles. 
        (Kind of equivalent to the env enviroment but for obstacles)
        
        Args:
            positions:    List of (x,y) initial centers for each obstacle.
            modes:        List of motion types, one per obstacle. Each must be one of
                          {"static","random","sinusoid","step_bounce","orbit"}.
            mode_params:  List of dicts (one per obstacle) specifying parameters:
                - step_bounce: {"bounds": (xmin,xmax), "speed": v, "dir": ±1}
                - orbit:       {"omega": omega, "center": (cx0,cy0)}
                - sinusoid:    {"amp": A, "freq": f, "phase": phi}
                - random:      {"sigma": sigma}
        """
        if not (len(positions) == len(modes) == len(mode_params)):
            raise ValueError(
                f"positions, modes, and mode_params must all have the same length, "
                f"but got {len(positions)}, {len(modes)}, and {len(mode_params)}."
            )
            
        self.m           = len(positions)
        self.modes       = modes
        #deepcopy to avoid modifying the original mode_params
        self.mode_params = copy.deepcopy(mode_params)

        #intializing states and parameters for velocity models
        self.vx = np.zeros(self.m)
        self.vy = np.zeros(self.m)
        
        self.cx = np.array([p[0] for p in positions], dtype=float)
        self.cy = np.array([p[1] for p in positions], dtype=float)
        
        self.dt = SAMPLING_TIME
        self.t  = 0.0
        
        # different types of movements that can be used
        self._step_functions = {
                "static"  : self._step_static,
                "random"  : self._step_random_walk,
                "sinusoid": self._step_sinusoid,
                "step_bounce": self._step_bounce,
                "orbit"      : self._step_orbit,
            }
        
        # intial state, will be used to restore the obstacles 
        # to their initial state when reset is called
        self._init_state = {
            "cx": self.cx.copy(),
            "cy": self.cy.copy(),
            "vx": self.vx.copy(),
            "vy": self.vy.copy(),
            "t":  self.t,
            "mode_params": copy.deepcopy(self.mode_params),
        }


    def step(self):
        """
        Advance the obstacle motions by one time step.

        For each obstacle i, calls the appropriate stepping method
        (static, random_walk, sinusoid, bounce, or orbit), updates
        its velocity, then updates its position.
        
         Returns:
            Updated (cx, cy) positions of all obstacles for one time step.
        """
        for i, mode in enumerate(self.modes):
            vx_i, vy_i = self._step_functions[mode](i)
            self.vx[i], self.vy[i] = vx_i, vy_i

        self.cx += - self.vx * self.dt
        self.cy += - self.vy * self.dt

        self.t += self.dt
        return self.cx.copy(), self.cy.copy()

    def _step_static(self, i: int):
        """
        Obstacle i remains fixed in place.
        
        Returns:
            Updated velocities (vx,vy) of obstacle i
        """
        return 0.0, 0.0

    def _step_random_walk(self, i: int):
        """
        Gaussian random-walk for obstacle i.
        
        
        Returns:
            Updated velocities (vx,vy) of obstacle i
        """
        sigma = self.mode_params[i].get("sigma", 0.1)
        vx_new = self.vx[i] + np.random.randn() * sigma
        vy_new = self.vy[i] + np.random.randn() * sigma
        return vx_new, vy_new

    def _step_sinusoid(self, i: int):
        """
        Sinusoidal motion for obstacle i.
        
        Returns:
            Updated velocities (vx,vy) of obstacle i
        """
        mp   = self.mode_params[i]
        amp  = mp.get("amp", 1.0)
        freq = mp.get("freq", 0.5)
        phase= mp.get("phase", 0.0)
        phi    = 2 * np.pi * freq * self.t + phase
        vx_i = amp * np.sin(phi)
        vy_i = amp * np.cos(phi)
        return vx_i, vy_i 
    
    def _step_bounce(self, i):
        """
        Step‐bounce motion along the x‐axis for obstacle i.

        - bounces between xmin and xmax
        - flips direction when hitting a bound
        
        Returns:
            Updated velocities (vx,vy) of obstacle i
        """
        mp     = self.mode_params[i]
        xmin, xmax = mp["bounds"]         
        speed = mp["speed"]                
        # read+update your current direction in-place:
        dir   = mp.get("dir", -1)          
        next_x = self.cx[i] - dir*speed*self.dt
        if next_x < xmin or next_x > xmax:
            dir *= -1
        mp["dir"] = dir                    # save flipped direction
        return dir*speed, 0.0
    
    def _step_orbit(self, i):
        """
        Rotate around a center point at angular rate omega.
        Velocity v = omega x r, where r is the distance to the center of rotation.
        
        Returns:
            Updated velocities (vx,vy) of obstacle i
        """
        mp     = self.mode_params[i]
        omega  = mp.get("omega", 1.0)            # rad/s
        cx0, cy0 = mp.get("center", (0.0, 0.0))  # pivot
        dx = self.cx[i] - cx0
        dy = self.cy[i] - cy0
        vx_i = -omega * dy
        vy_i =  omega * dx
        return vx_i, vy_i
    
    def current_positions(self):
        return list(zip(self.cx, self.cy))
    
    def predict_states(self, N: int):
        """
        Simulate N steps ahead (without committing) and return the
        predicted trajectories for x and y positions.
        
        After predicting, restores the original obstacle state.
        
        Args: 
            N : Number of future steps to simulate. [integer]
        
        Returns:
            x_pred_flat : np.ndarray, shape ((N+1)*m,)
            y_pred_flat : np.ndarray, shape ((N+1)*m,)
        
        """
        prediction = {
            "cx": self.cx.copy(),
            "cy": self.cy.copy(),
            "vx": self.vx.copy(),
            "vy": self.vy.copy(),
            "t":  self.t,
            "mode_params": copy.deepcopy(self.mode_params),
        }

        # simulate N steps
        x_pred = np.zeros((N+1, self.m))
        y_pred = np.zeros((N+1, self.m))
        
        x_pred[0, :] = self.cx
        y_pred[0, :] = self.cy
        
        for k in range(1,N+1):
            x_k, y_k = self.step()
            x_pred[k, :] = x_k
            y_pred[k, :] = y_k
            
            
        # restore from backup
        self.cx          = prediction["cx"]
        self.cy          = prediction["cy"]
        self.vx          = prediction["vx"]
        self.vy          = prediction["vy"]
        self.t           = prediction["t"]
        self.mode_params = prediction["mode_params"]

        return x_pred.flatten(), y_pred.flatten()
    
    def reset(self):
        """
        Restore intial state of the obstacles.
        
        Returns:
            Reset positions, velocities and time of all obstacles.
        """
        st = self._init_state
        self.cx          = st["cx"].copy()
        self.cy          = st["cy"].copy()
        self.vx          = st["vx"].copy()
        self.vy          = st["vy"].copy()
        self.t           = st["t"]
        self.mode_params = copy.deepcopy(st["mode_params"])
        return self.current_positions()
    
    
class RNN:
    def __init__(self, layers_list, positions, radii, horizon):
        #TODO: Finish commenting and cleaning this funciton and onwards
        """
        Build an Elman-style RNN for CBF-based MPC.

        Parameters
        ----------
        layers_list : list[int]
            Network architecture as [input_dim, hidden_dim1, ..., output_dim].
        positions : list[tuple]
            List of (x, y) centers for obstacles (used by CBF input functions).
        radii : list[float]
            Obstacle radii corresponding to positions.
        horizon : int
            Number of time steps to unroll the RNN in MPC.

        Attributes
        ----------
        obst : Obstacles
            Obstacles object containing positions and radii.
        layers_list : list --> layers_list = [input_dim, hidden_dim, ..., output_dim]
            Stored network dimensions.
        rnn_weights_ih : list[cs.MX]
            Input-to-hidden weight symbols per layer.
        rnn_weights_hh : list[cs.MX]
            Hidden-to-hidden recurrent weight symbols (for all but last layer).
        rnn_biases_ih : list[cs.MX]
            Bias symbols per layer.
        hidden_sym_list : list[cs.MX]
            Initial hidden-state symbols for each layer (except output layer).
        input_sym : cs.MX
            Symbolic placeholder for the entire input sequence (input_dim (state,hx) x horizon).
        activations : list
            List of activation functions per layer.
        horizon : int
            Horizon length of MPC and accordingly also for the RNN.
        """
        self.obst         = Obstacles(positions, radii)
        self.layers_list  = layers_list
        # four lists of CasADi symbols, one per Rrnn layer:
        self.rnn_weights_ih = []
        self.rnn_weights_hh = []
        self.rnn_biases_ih  = []
        self.hidden_sym_list = []
        # Sequence input for multi-step forward (used in forward_rnn)
        self.input_sym = cs.MX.sym("x", self.layers_list[0], horizon)  # input vector
        self.activations = []
        self._build_network()
        self.np_random = np.random.default_rng(seed=SEED)
        self.horizon = horizon
        
        self.positions = positions
        self.radii     = radii
            

    def relu(self, x):
        """Standard ReLU: max(x, 0)."""
        return cs.fmax(x, 0)
    
    def tanh(self, x):
        """Hyperbolic tangent activation."""
        return cs.tanh(x)
    
    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU to avoid dead neurons."""
        return cs.fmax(x, 0) + alpha * cs.fmin(x, 0)
    
    def shifted_sigmoid(self, x, epsilon=1e-6):
        """
        Sigmoid shifted into (epsilon, 1) since we cant have a 0
        """
        return epsilon + (1 - epsilon) / (1 + cs.exp(-x))
    
    def normalization_z(self, rnn_input):
        
        """
        Normalizes based on maximum and minimum values of the states and h(x) values.

        """
  
        # x_norm = (nn_input[:4] - mu_states) / sigma_states
        
        x_min  = cs.DM([-CONSTRAINTS_X[0], -CONSTRAINTS_X[0], -CONSTRAINTS_X[0], -CONSTRAINTS_X[0]]) # minimum values of the states
        x_max = cs.DM([0, 0, 0, 0]) # maximum values of the states
        
        x_norm = (rnn_input[:4]-x_min)/(x_max-x_min) # normalize the states based on the maximum values
        
        h_max_list = []
        for (px, py), r in zip(self.positions, self.radii):
            dx = x_min[0] - px
            dy = x_min[1] - py
            h_max_i = dx**2 + dy**2 - r**2
            h_max_list.append(h_max_i)
        
        h_raw = rnn_input[4:]             
        h_norm_list = []
        
        h_raw = cs.reshape(h_raw, -1, 1)  # reshape to (m, 1) where m is the number of obstacles

        h_raw_split = cs.vertsplit(h_raw)  # returns a list of (1x1) MX elements

        for i, h_i in enumerate(h_raw_split):
            h_max_i = h_max_list[i]
            h_norm_i = h_i / h_max_i
            h_norm_i = cs.fmin(cs.fmax(h_norm_i, 0), 1)
            h_norm_list.append(h_norm_i)
            
        h_norm = cs.vertcat(*h_norm_list)
        return cs.vertcat(x_norm, h_norm)

    # ─── low‐level linear + cell ─────────────────────────────────────────────────
    def linear(self, inp, weight, bias=None):
        
        """
        Compute a linear transform: out = weight @ inp + bias.

        Parameters
        ----------
        inp : cs.MX or cs.DM
            Input vector of shape (input_dim x 1).
        weight : cs.MX
            Weight matrix of shape (output_dim x input_dim).
        bias : cs.MX, optional
            Bias vector of shape (output_dim x 1). If None, no bias added.

        Returns
        -------
        cs.MX
            Output of linear transform (output_dim x 1).
        """
        
        out = weight @ inp
        if bias is not None:
            out = out + bias
        return out

    def rnn_cell(self, inp, hidden, Wih, Whh, bih, activation):
        """
        Combination of hidden and input to compute the new output and hidden state.

        In short it Computes:
            h_new = activation(Wih @ inp + bih + Whh @ h_prev)

        Parameters
        ----------
        inp : cs.MX or cs.DM
            Current input (input_dim x 1).
        hidden : cs.MX or cs.DM or None
            Previous hidden state (output_dim x 1). If None, initialized to zeros.
        Wih : cs.MX
            Input-to-hidden weights (output_dim x input_dim).
        Whh : cs.MX
            Hiddenxtoxhidden weights (output_dim x output_dim).
        bih : cs.MX
            Bias for input-to-hidden (output_dim x 1).
        activation : the activation function
            Element-wise activation function.

        Returns
        -------
        cs.MX
            the output (output_dim x 1).
        """
        if hidden is None:
            hidden = cs.DM.zeros(Wih.shape[0], 1)
            
        pre = self.linear(inp, Wih, bih) + self.linear(hidden, Whh)
        return activation(pre)


    def make_rnn_step(self):
        """
        Build a CasADi Function for one RNN step.

        Returns
        -------
        cs.Function
            `Rstep(h_current, ..., h_current_K, (x, h_cbf(x_t))= x_t,
                   Wih_0, bih_0, Whh_0, ..., Wih_{L}, bih_{L})`
            → `(h_next_1, ..., h_next_K, y_t)`.

        Inputs
        ------
        h_current                     : MX, shape (hidden_dim_i x 1)
        x_t = (x, h_cbf(x_t))        : MX, shape (input_dim x 1)
        Wih_i                        : MX, shape (hidden_dim_i x input_dim_i)
        bih_i                        : MX, shape (hidden_dim_i x 1)
        Whh_i                        : MX, shape (hidden_dim_i x hidden_dim_i) (for i < output layer)

        Outputs
        -------
        h_next    : MX, updated hidden state for next layer
        y_t        : MX, final-layer output (output_dim x 1)
        """
        
        L        = len(self.rnn_weights_ih) # number of RNN layers\
            
            
        print(f"L:{L}")    
        input_sym = cs.MX.sym("x", self.layers_list[0],1)  # input vector 
        params   = []
        next_hid_list = []
        
        for i in range(L-1):
            params += [
                self.rnn_weights_ih[i],
                self.rnn_biases_ih [i],
                self.rnn_weights_hh[i],
            ]
        # now the last layer only has Wih & bih:
        params += [
        self.rnn_weights_ih[L-1],
        self.rnn_biases_ih [L-1],
            ]
                  
        out_int = input_sym
        for i in range(L):
            if i<L-1:
                out_int = self.rnn_cell(
                    out_int, self.hidden_sym_list[i],
                    self.rnn_weights_ih[i],
                    self.rnn_weights_hh[i],
                    self.rnn_biases_ih[i],
                    self.activations[i]
                )
                next_hid_list.append(out_int)
            else:
                out_int = self.rnn_cell(
                    out_int, None,
                    self.rnn_weights_ih[i],
                    self.Whh0,  # no Whh for last layer
                    self.rnn_biases_ih[i],
                    self.activations[i]
                )
        y_out = out_int   # row‐vector output
    
        
        return cs.Function("Rstep", [*self.hidden_sym_list, input_sym, *params], [*next_hid_list, y_out])
    
    
    def forward_rnn(self):
        """
        Unroll the RNN for horizon steps.

        Returns
        -------
        cs.Function
            rnn_forward(flat_input, h0_1,...,h0_K, Wih_0,bih_0,Whh_0,...,Wih_L,bih_L)
            → (H_stack, Y_stack).

        Inputs
        ------
        flat_input : MX, shape ((input_dim + m) * horizon × 1)
            Stacked [x0; h_obs(x0); x1; h_obs(x1); ...].
        h0_i       : MX, initial hidden states.
        Wih_i, bih_i, Whh_i : MX, as in `make_rnn_step`.

        Outputs
        -------
        H_stack    : MX, all hidden states over time, shape (sum(hidden_dims)*horizon × 1)
        Y_stack    : MX, all final‐layer outputs over time, shape (output_dim*horizon × 1)
        """

        rnn_step = self.make_rnn_step()

        h = self.hidden_sym_list   # MX
        params = self.get_flat_parameters_list()  

        h_history = []
        y_history = []
        
        

        for i in range(self.horizon):
            x_t_raw = self.input_sym[:, i]
            x_t = self.normalization_z(x_t_raw)
            *h, y = rnn_step(*h, x_t, *params)
            h_history.append(cs.vertcat(*h))
            y_history.append(y)

        H_stack = cs.vertcat(*h_history)   # (hidden_dim * horizon)×1
        Y_stack = cs.vertcat(*y_history)   # (output_dim * horizon)×1
        
        flat_input_sym = cs.reshape(self.input_sym, -1, 1)

        return cs.Function(
        "rnn_forward",
        [flat_input_sym, *self.hidden_sym_list, *params],
        [H_stack, Y_stack],
        {"cse": True}
        )

    def _build_network(self):
            """
            Create symbolic parameters & initial-state symbols for each layer:
            - For layers 0..L-2: Wih, bih, Whh, hidden_sym
            - For last layer:     Wih, bih; Whh0 := zeros
            Also activations (leaky_relu and shifted_sigmoid).
            """
            for i in range(len(self.layers_list) - 1):

                in_dim  = self.layers_list[i]
                hid_dim = self.layers_list[i+1]

                Wih = cs.MX.sym(f"Wih{i}", hid_dim, in_dim)
                bih = cs.MX.sym(f"bih{i}", hid_dim, 1)
                
                self.rnn_weights_ih.append(Wih)
                self.rnn_biases_ih.append(bih)

                if i != (len(self.layers_list) - 2):  # not the last layer
                    hidden_sym = cs.MX.sym(f"hinit{i}", hid_dim, 1)
                    Whh = cs.MX.sym(f"Whh{i}", hid_dim, hid_dim)
                    self.rnn_weights_hh.append(Whh)
                    self.hidden_sym_list.append(hidden_sym)
                else:
                    # Whh0 = cs.MX.sym(f"Whhzero", self.layers_list[-1], self.layers_list[-1])
                    self.Whh0 = cs.DM.zeros(self.layers_list[-1], self.layers_list[-1])
    
                if i == len(self.layers_list) - 2:
                    self.activations.append(self.shifted_sigmoid)
                else:
                    self.activations.append(self.leaky_relu)
                    
                    
    def initialize_parameters(self):
        """
        He-uniform init to match exactly get_flat_parameters() ordering.
        Returns
        -------
        flat_params : casadi.DM, shape=(410,1)
            All Wih, bih, Whh (where applicable) stacked exactly as get_flat_parameters().
        Wih_vals    : list of numpy.ndarray
            The fan-in/fan-out weight matrices.
        Whh_vals    : list of numpy.ndarray
            The recurrent weight matrices (one per non-last layer).
        bih_vals    : list of numpy.ndarray
            The input-to-hidden biases (one per layer).
        """
        L = len(self.layers_list) - 1  # number of layers
        Wih_vals = []
        Whh_vals = []
        bih_vals = []

    
        for i in range(L):
            fan_in, fan_out = self.layers_list[i], self.layers_list[i+1]

            if i < L-1:
            # bound1 = 3*np.sqrt(6.0 / (fan_in + fan_out))
                bound_low = np.sqrt(6.0 / fan_in)
                bound_high = np.sqrt(6.0 / fan_out)
                
                Wih_v = 0.1*self.np_random.uniform(low=-bound_low, high=bound_high, size=(fan_out, fan_in)) #self.np_random.uniform(-bound1, bound1, size=(fan_out, fan_in))
                Wih_vals.append(Wih_v)

                Whh_v = 0.1*self.np_random.uniform(low=-bound_low, high=bound_high, size=(fan_out, fan_out))
                Whh_vals.append(Whh_v)
            
            else:
                bound = 0.5*np.sqrt(6.0 / (fan_out + fan_out))
                Wih_v = self.np_random.uniform(-bound, bound, size=(fan_out, fan_in))
                Wih_vals.append(Wih_v)
            
            bih_v = np.zeros((fan_out, 1))
            bih_vals.append(bih_v)
                
        raws = []

        for i in range(L-1):
            raws.append(Wih_vals[i].reshape(-1, 1))
            raws.append(bih_vals[i].reshape(-1, 1))
            raws.append(Whh_vals[i].reshape(-1, 1))

        raws.append(Wih_vals[L-1].reshape(-1, 1))
        raws.append(bih_vals[L-1].reshape(-1, 1))

  
        flat = cs.vertcat(*[cs.DM(r) for r in raws])
        return flat, Wih_vals, Whh_vals, bih_vals

    def get_flat_parameters_list(self):
        """
       for mapaccum call
        """
        out = []
        L = len(self.layers_list) - 1  
 
        for i in range(L-1):
            out += [
                self.rnn_weights_ih[i],
                self.rnn_biases_ih [i],
                self.rnn_weights_hh[i],
            ]
     
        out += [
            self.rnn_weights_ih[L-1],
            self.rnn_biases_ih [L-1],
        ]
        return out

    def get_flat_parameters(self):
        """
        for NLP p-argument.
        """
        raws = []
        L = len(self.layers_list) - 1
      
        for i in range(L-1):
            raws += [
                cs.reshape(self.rnn_weights_ih[i], -1, 1),
                cs.reshape(self.rnn_biases_ih [i], -1, 1),
                cs.reshape(self.rnn_weights_hh[i], -1, 1),
            ]
 
        raws += [
            cs.reshape(self.rnn_weights_ih[L-1], -1, 1),
            cs.reshape(self.rnn_biases_ih [L-1], -1, 1),
        ]
        return cs.vertcat(*raws)
    
    def unpack_flat_parameters(self, flat_params):
        """
        Given a (Px1) vector flat_params in the same ordering as get_flat_parameters(),
        return a Python list [Wih0, bih0, Whh0, Wih1, bih1, Whh1, …, Wih_{L-1}, bih_{L-1}].
        Done so we turn params["rnn_params"] into a list of matrices as set up accordingly for this class.
        """
        # Number of layers
        L = len(self.layers_list) - 1

        # Precompute fan‐in/out dims
        dims = [(self.layers_list[i], self.layers_list[i+1]) for i in range(L)]

        idx = 0
        unpacked = []

        for i, (fan_in, fan_out) in enumerate(dims):
            # Wih_i has shape (fan_out, fan_in)
            num_Wih = fan_out * fan_in
            Wih_i = cs.reshape(flat_params[idx:idx+num_Wih], fan_out, fan_in)
            unpacked.append(Wih_i)
            idx += num_Wih

            # bih_i has shape (fan_out, 1)
            num_bih = fan_out
            bih_i = cs.reshape(flat_params[idx:idx+num_bih], fan_out, 1)
            unpacked.append(bih_i)
            idx += num_bih

            # Only non‐last layers have a Whh
            if i < L-1:
                num_Whh = fan_out * fan_out
                Whh_i = cs.reshape(flat_params[idx:idx+num_Whh], fan_out, fan_out)
                unpacked.append(Whh_i)
                idx += num_Whh

        return unpacked


class env(gym.Env):

    def __init__(self):
        super().__init__()
        self.na = NUM_INPUTS
        self.ns = NUM_STATES
        self.umax = CONSTRAINTS_U
        self.umin = -CONSTRAINTS_U

        self.observation_space = Box(-CONSTRAINTS_X[0], CONSTRAINTS_X[1], (self.ns,), np.float64)
        self.action_space = Box(self.umin, self.umax, (self.na,), np.float64)
        self.dt = SAMPLING_TIME

        self.A = np.array([
            [1, 0, self.dt, 0], 
            [0, 1, 0, self.dt], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ])


    def reset(self, seed, options):
        super().reset(seed=seed, options=options)
        self.x = np.asarray([-CONSTRAINTS_X[0], -CONSTRAINTS_X[1], 0, 0])
        # assert self.observation_space.contains(self.x), f"invalid reset state {self.x}"
        return self.x, {}

    def step(
        self, action):
        # x = self.x
        u = np.asarray(action).reshape(self.na)
        self.B = np.array([
            [0.5 * self.dt**2, 0], 
            [0, 0.5 * self.dt**2], 
            [self.dt, 0], 
            [0, self.dt]
        ])
        # assert self.action_space.contains(u), f"invalid action {u}"

        x_new = self.A @ self.x + self.B @ u
        # assert self.observation_space.contains(x_new), f"invalid new state {x_new}"
        self.x = x_new
        return x_new, np.nan, False, False, {}


class MPC:
    # constant of MPC class
    ns = NUM_STATES # num of states
    na = NUM_INPUTS # num of inputs

    def __init__(self, layers_list, horizon, positions, radii, slack_penalty):
        """
        Initialize the MPC class with parameters.
        
        Initialize the MPC problem:
         - build discrete-time system matrices A, B
         - set up CasADi symbols for Q, R, P, V0
         - instantiate the RNN for the CBF
         - prepare CasADi decision variables X, U, S
         - build dynamics function f(x,u) and RNN forward pass
        
        """
        self.ns = MPC.ns
        self.na = MPC.na
        self.horizon = horizon


        dt = SAMPLING_TIME

        # discrete‐time dynamics matrices
        self.A = np.array([
            [1, 0, dt, 0], 
            [0, 1, 0, dt], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ])

        self.B = np.array([
            [0.5 * dt**2, 0], 
            [0, 0.5 * dt**2], 
            [dt, 0], 
            [0, dt]
        ])

        # fixed state‐cost weight
        self.Q = np.diag([10, 10, 10, 10])
        
        # symbolic parameters for MPC
        self.A_sym = cs.MX.sym("A", self.ns, self.ns)
        self.B_sym = cs.MX.sym("B", self.ns, self.na)
        self.b_sym = cs.MX.sym("b", self.ns)

        self.P_diag = cs.MX.sym("P_diag", self.ns, 1)
        self.P_sym = cs.diag(self.P_diag)

        self.Q_sym = cs.MX.sym("Q", self.ns, self.ns)
        self.R_sym = cs.MX.sym("R", self.na, self.na)
        self.V_sym = cs.MX.sym("V0")
        
        # instantiate the RNN
        self.rnn = RNN(layers_list, positions, radii, horizon)
        self.m = self.rnn.obst.obstacle_num  # number of obstacles
        
        # predicted obstacle velocities over horizon
        self.xpred_list = cs.MX.sym("xpred_list", self.m)  # velocity in x direction
        self.ypred_list = cs.MX.sym("ypred_list", self.m) # velocity in y direction
        
        # flattened predictions for each time step
        self.xpred_hor = cs.MX.sym("xpred_hor", self.m * (self.horizon+1))
        self.ypred_hor = cs.MX.sym("ypred_hor", self.m * (self.horizon+1))
        
        # weight on slack variables in CBF constraints
        self.weight_cbf = cs.DM([slack_penalty])


        # decision variables:
        #   X_sym: states over horizon+1
        #   U_sym: control inputs over horizon
        #   S_sym: slack for CBF constraints (m × horizon)
        self.X_sym = cs.MX.sym("X", self.ns, self.horizon+1)
        self.U_sym = cs.MX.sym("U",self.na, self.horizon)
        self.S_sym = cs.MX.sym("S", self.m, self.horizon)

        # states for one time step and inputs for one time step
        self.x_sym = cs.MX.sym("x", MPC.ns)
        self.u_sym = cs.MX.sym("u", MPC.na)


        # CasADi function for one‐step dynamics: f(x,u) = A x + B u
        x_new  = self.A @ self.x_sym + self.B @ self.u_sym 
        self.dynamics_f = cs.Function('f', [self.x_sym, self.u_sym], [x_new], ['x','u'], ['ode'])

    
        # CBF h‐functions for each obstacle
        self.h_funcs = self.rnn.obst.make_h_functions()  # length‐m Python list of casadi.Function

        # build flat‐input function for RNN
        flat_input_fn =self.make_flat_input_fn()
        flat_input = flat_input_fn(self.X_sym, self.xpred_hor, self.ypred_hor)

        # hidden‐state symbols and parameter symbols for the RNN
        self.hid_syms = self.rnn.hidden_sym_list            # [h0_layer0, h0_layer1, h0_layer2]
        param_syms = self.rnn.get_flat_parameters_list()  # [Wih0, bih0, Whh0, Wih1, …, bih3]
        
        # RNN forward‐pass over the horizon to get alphas
        self.rnn_fwd_func = self.rnn.forward_rnn()
        #Y_stack containts list of alphas: [alpha_0, alpha_1, ..., alpha_{m*horizon}]
        _, Y_stack = self.rnn_fwd_func(flat_input,
            *self.hid_syms,    # 3 separate MX args   
            *param_syms   # 11 separate MX args
        )

        self.alpha_list = Y_stack

    def state_const(self):

        """
        Build linear dynamics constraints:
          X_{k+1} - [A_sym X_k + B_sym U_k + b_sym] == 0, for k=0…horizon
        """

        state_const_list = []

        for k in range(self.horizon):

            state_const_list.append( self.X_sym[:,k+1] - ( self.A_sym @ self.X_sym[:,k] + self.B_sym @ self.U_sym[:,k] + self.b_sym ) )

        self.state_const_list = cs.vertcat( *state_const_list )
        
        print(f"self.state_const_list shape: {self.state_const_list.shape}")

        return 
    
    def make_flat_input_fn(self):
        """
        Returns a CasADi Function that maps:
           (X, xpred_hor, ypred_hor)
           -->flattened sequence: [input[0], input[1], ..., input[horizon]]
           where input at time step k: input[k] = [x_k; h_i(x_k, xpred_hor[k*m:(k+1)*m], ypred_hor[k*m:(k+1)*m]) for i in range(m)]
        """
        X = cs.MX.sym("X", self.ns, self.horizon+1)

        inter = []
        for t in range(self.horizon):
            x_t    = X[:, t]
            cbf_t  = [h_i(x_t, 
                          self.xpred_hor[t*self.m:(t+1)*self.m], 
                          self.ypred_hor[t*self.m:(t+1)*self.m]) 
                      for h_i in self.h_funcs]           # m×1 each
            inter.append(x_t)                            # ns×1
            inter.extend(cbf_t)                          # m scalars

        flat_in = cs.vertcat(*inter)  # ((ns+m)*horizon)×1
        return cs.Function("flat_input", [X, self.xpred_hor, self.ypred_hor], [flat_in], ["X",  "xpred_list", "ypred_list"], ["flat_in"])
    
    
    def cbf_const(self):
        """
        Build the slack-augmented CBF constraints phi_i + S >= 0
        for each i=1...m and each time k=0...h-1:
        phi_i = h_i(x_{k+1}) - h_i(x_k) + alpha_k*_i(x_k)
        """
        cons = []
        m = len(self.h_funcs)

        for k in range(self.horizon):
            xk = self.X_sym[:, k]      # 4×1
            uk = self.U_sym[:, k]      # 2×1

            phi_k_list = []
            for i, h_i in enumerate(self.h_funcs):

                alpha_ki = self.alpha_list[k*m + i]        # MX-scalar

                h_x     = h_i(xk, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m])     # h(x_k)
                x_next  = self.dynamics_f(xk, uk)          # f(x_k, u_k)
                h_xnext = h_i(x_next, self.xpred_hor[(k+1)*self.m:(k+2)*self.m], self.ypred_hor[(k+1)*self.m:(k+2)*self.m])                      # h(x_{k+1})

                # phi_i = h(x_{k+1}, k+1) − h(x_k, k) + alpha*h(x_k, k) TIME DEPENDENT h()
                phi_i = h_xnext - h_x + alpha_ki * h_x
                phi_k_list.append(phi_i)

            # now add slack
            phi_k = cs.vertcat(*phi_k_list)                # m×1
            cons.append(phi_k + self.S_sym[:, k])         # m×1

        # final (m*horizon)×1 vector
        self.cbf_const_list = cs.vertcat(*cons)
   
    def cbf_const_noslack(self):
        """
        Same as cbf_const but without slack variables.
        """
        cons = []
        m = len(self.h_funcs)

        for k in range(self.horizon):
            xk = self.X_sym[:, k]      # 4×1
            uk = self.U_sym[:, k]      # 2×1
            phi_k_list = []
            for i, h_i in enumerate(self.h_funcs):
                
                alpha_ki = self.alpha_list[k*m + i]        # MX-scalar


                h_x     = h_i(xk, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m])              # h(x_k)
                x_next  = self.dynamics_f(xk, uk)          # f(x_k, u_k)
                h_xnext = h_i(x_next, self.xpred_hor[(k+1)*self.m:(k+2)*self.m], self.ypred_hor[(k+1)*self.m:(k+2)*self.m])                       # h(x_{k+1})

                # phi_i = h(x_{k+1}, k+1) − h(x_k, k) + alpha*h(x_k, k) TIME DEPENDENT h()
                phi_i = h_xnext - h_x + alpha_ki * h_x
                phi_k_list.append(phi_i)

            phi_k = cs.vertcat(*phi_k_list)                # m×1
            cons.append(phi_k)         # m×1

        # final (m*horizon)×1 vector
        self.cbf_const_list_noslack = cs.vertcat(*cons)
    
    def objective_method(self):

        """""
        Builds MPC stage cost and terminal cost
        stage cost: sum_{k,i} [ x.T @ Q @ x + u.T @ R @ u + weight_cbf * S_i]
        terminal cost: x.T @ P @ x
        """
        stage_cost = sum(
            (self.X_sym[:, k].T @ self.Q_sym @ self.X_sym[:, k] + 
            self.U_sym[:, k].T @ self.R_sym @ self.U_sym[:, k] + 
            self.weight_cbf * self.S_sym[m,k])
            for m in range (self.m) for k in range(self.horizon)
        )
        terminal_cost = cs.bilin((self.P_sym), self.X_sym[:, -1])
        self.objective = self.V_sym + terminal_cost + stage_cost

        return
    
    def objective_method_noslack(self):

        """""
        stage cost calculation
        """
        # why doesnt work? --> idk made it in line
        stage_cost = sum(
            (self.X_sym[:, k].T @ self.Q_sym @ self.X_sym[:, k] + 
            self.U_sym[:, k].T @ self.R_sym @ self.U_sym[:, k])
            for k in range(self.horizon)
        )
        #slack penalty
        terminal_cost = cs.bilin((self.P_sym), self.X_sym[:, -1])


        self.objective_noslack = self.V_sym + terminal_cost + stage_cost

        return
    
    def MPC_solver_noslack(self):
        """""
        Create and return a CasADi NLP solver for MPC without slack.
        MPC built according to V-value function setup
        """
        self.state_const()
        self.objective_method_noslack()
        self.cbf_const_noslack()

        # Flatten matrices to put in as vector
        X_flat = cs.reshape(self.X_sym, -1, 1) 
        U_flat = cs.reshape(self.U_sym, -1, 1)  

        A_sym_flat = cs.reshape(self.A_sym , -1, 1)
        B_sym_flat = cs.reshape(self.B_sym , -1, 1)
        Q_sym_flat = cs.reshape(self.Q_sym , -1, 1)
        R_sym_flat = cs.reshape(self.R_sym , -1, 1)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat),
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, self.P_diag, Q_sym_flat, 
                            R_sym_flat, self.rnn.get_flat_parameters(), self.xpred_hor, self.ypred_hor, *self.hid_syms),
            "f": self.objective_noslack, 
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list_noslack),
        }

        opts = {
            # (Replace MX with SX expressions in problem formulation)
            # Automatically expand and simplify symbolic expressions before solving
            "expand": True,
            # 	print information about execution time
            "print_time": False,
            # 	Ensure that primal-dual solution is consistent with the bounds (aka you make sure bound 0 is 0 and not 1e-8)
            "bound_consistency":True,
            # Calculate Lagrange multipliers
            "calc_lam_x": True,
            "calc_lam_p": True,
            "calc_multipliers": True,
            # When errors occur during evaluation of f,g,...,stop the iterations
            "eval_errors_fatal": True,
            # Throw exceptions when function evaluation fails (default true).
            "error_on_fail": False,
            "ipopt": {"max_iter": 500, "print_level": 0, "warm_start_init_point": "yes"},
        }

        # Create the NLP solver instance
        MPC_solver = cs.nlpsol("solver", "ipopt", nlp, opts)
        return MPC_solver
    
    def MPC_solver(self):
        """""
        solves the MPC according to V-value function setup
        """
        self.state_const()
        self.objective_method()
        self.cbf_const()

        # Flatten matrices to put in as vector
        X_flat = cs.reshape(self.X_sym, -1, 1) 
        S_flat = cs.reshape(self.S_sym, -1, 1)  
        U_flat = cs.reshape(self.U_sym, -1, 1)  

        A_sym_flat = cs.reshape(self.A_sym , -1, 1)
        B_sym_flat = cs.reshape(self.B_sym , -1, 1)
        P_sym_flat = cs.reshape(self.P_sym , -1, 1)
        Q_sym_flat = cs.reshape(self.Q_sym , -1, 1)
        R_sym_flat = cs.reshape(self.R_sym , -1, 1)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat, S_flat),
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, self.P_diag, Q_sym_flat, 
                            R_sym_flat, self.rnn.get_flat_parameters(), self.xpred_hor, self.ypred_hor, *self.hid_syms),
            "f": self.objective, 
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list),
        }

        opts = {
            # (Replace MX with SX expressions in problem formulation)
            # Automatically expand and simplify symbolic expressions before solving
            "expand": True,
            # 	print information about execution time
            "print_time": False,
            # 	Ensure that primal-dual solution is consistent with the bounds (aka you make sure bound 0 is 0 and not 1e-8)
            "bound_consistency":True,
            # Calculate Lagrange multipliers
            "calc_lam_x": True,
            "calc_lam_p": True,
            "calc_multipliers": True,
            # When errors occur during evaluation of f,g,...,stop the iterations
            "eval_errors_fatal": True,
            # Throw exceptions when function evaluation fails (default true).
            "error_on_fail": False,
            "ipopt": {"max_iter": 500, "print_level": 0, "warm_start_init_point": "yes"},
        }

        MPC_solver = cs.nlpsol("solver", "ipopt", nlp, opts)

        return MPC_solver
    

    def MPC_solver_rand(self):
        """""
        solves the MPC according to V-value function setup
        """
        self.state_const()
        self.objective_method()
        self.cbf_const()

        # Flatten matrices to put in as vector
        X_flat = cs.reshape(self.X_sym, -1, 1)
        S_flat = cs.reshape(self.S_sym, -1, 1)    
        U_flat = cs.reshape(self.U_sym, -1, 1)


        A_sym_flat = cs.reshape(self.A_sym , -1, 1)
        B_sym_flat = cs.reshape(self.B_sym , -1, 1)
        P_sym_flat = cs.reshape(self.P_sym , -1, 1)
        Q_sym_flat = cs.reshape(self.Q_sym , -1, 1)
        R_sym_flat = cs.reshape(self.R_sym , -1, 1)

        rand_noise = cs.MX.sym("rand_noise", 2)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat, S_flat),
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, self.P_diag, Q_sym_flat, R_sym_flat, 
                            self.rnn.get_flat_parameters(), rand_noise, self.xpred_hor, self.ypred_hor, *self.hid_syms),
            "f": self.objective + rand_noise.T @ self.U_sym[:,0], 
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list),
        }

        opts = {
            # (Replace MX with SX expressions in problem formulation)
            # Automatically expand and simplify symbolic expressions before solving
            "expand": True,
            # 	print information about execution time
            "print_time": False,
            # 	Ensure that primal-dual solution is consistent with the bounds (aka you make sure bound 0 is 0 and not 1e-8)
            "bound_consistency":True,
            # Calculate Lagrange multipliers
            "calc_lam_x": True,
            "calc_lam_p": True,
            "calc_multipliers": True,
            # When errors occur during evaluation of f,g,...,stop the iterations
            "eval_errors_fatal": True,
            # Throw exceptions when function evaluation fails (default true).
            "error_on_fail": False,
            "ipopt": {"max_iter": 500, "print_level": 0, "warm_start_init_point": "yes"},
        }

        MPC_solver = cs.nlpsol("solver", "ipopt", nlp, opts)

        return MPC_solver
    
    def generate_symbolic_mpcq_lagrange(self):
        """
          Construct a CasADi Function that computes the gradient of the MPC Lagrangian
        """
        self.state_const()
        self.objective_method()

        X_flat = cs.reshape(self.X_sym, -1, 1)  # Flatten 
        U_flat = cs.reshape(self.U_sym, -1, 1)
        S_flat = cs.reshape(self.S_sym, -1, 1)   

        opt_solution = cs.vertcat(X_flat, U_flat, S_flat)

      
        # X_con + U_con + S_con 
        lagrange_mult_x_lb_sym = cs.MX.sym("lagrange_mult_x_lb_sym", self.ns * (self.horizon+1) + 
                                           self.na * (self.horizon) + self.rnn.obst.obstacle_num * (self.horizon))
        lagrange_mult_x_ub_sym = cs.MX.sym("lagrange_mult_x_ub_sym", self.ns * (self.horizon+1) + 
                                           self.na * (self.horizon) + self.rnn.obst.obstacle_num  * (self.horizon))
        lagrange_mult_g_sym = cs.MX.sym("lagrange_mult_g_sym", 1*self.ns*(self.horizon) + 
                                        self.rnn.obst.obstacle_num*self.horizon)


        X_lower_bound = -np.tile(CONSTRAINTS_X, self.horizon)
        X_upper_bound = np.tile(CONSTRAINTS_X, self.horizon)

        U_lower_bound = -np.ones(self.na * (self.horizon-1))
        U_upper_bound = np.ones(self.na * (self.horizon-1))  

        # X_con + U_con + S_con + Sx_con
        lbx = cs.vertcat(self.X_sym[:,0], cs.DM(X_lower_bound), self.U_sym[:,0], cs.DM(U_lower_bound), 
                         np.zeros(self.rnn.obst.obstacle_num *self.horizon)) 
        ubx = cs.vertcat(self.X_sym[:,0], cs.DM(X_upper_bound), self.U_sym[:,0], cs.DM(U_upper_bound), 
                         np.inf*np.ones(self.rnn.obst.obstacle_num *self.horizon))

        # construct lower bound here 
        lagrange1 = lagrange_mult_x_lb_sym.T @ (opt_solution - lbx) #positive @ negative
        lagrange2 = lagrange_mult_x_ub_sym.T @ (ubx - opt_solution)  # positive @ negative                                
        lagrange3 = lagrange_mult_g_sym.T @ cs.vertcat(self.state_const_list, -self.cbf_const_list) 
        #SHOULD I HAVE self.cbf_const_list be negative here?
        #no it was already g(x)=<0
        # means lag multiplier is lambda>=0
        # so we get lambda.T@g(x)
        # if self.cbf_const_list didnt have a minus infront it would be g(x)>=0, but the according lagrangians wouldnt hold
        
 
        # theta_vector = cs.vertcat(self.P_diag, self.rnn.get_flat_parameters())
        theta_vector = cs.vertcat(self.rnn.get_flat_parameters())

        self.theta = theta_vector

        qlagrange = self.objective + lagrange1 + lagrange2 + lagrange3

        #computing derivative of lagrangian for A
        _, qlagrange_sens = cs.hessian(qlagrange, theta_vector)

        qlagrange_fn = cs.Function(
            "qlagrange_fn",
            [
                self.A_sym, self.B_sym, self.b_sym, self.Q_sym, self.R_sym,
                self.P_sym, lagrange_mult_x_lb_sym, lagrange_mult_x_ub_sym, 
                lagrange_mult_g_sym, X_flat, U_flat, S_flat, self.rnn.get_flat_parameters(),
                self.xpred_hor, self.ypred_hor, *self.hid_syms
            ],
            [qlagrange_sens]
        )

        return qlagrange_fn, _
    
    def qp_solver_fn(self):
        """
          Construct and return a small QP solver (OSQP) for constrained parameter updates.
        """

        Hessian_sym = cs.MX.sym("Hessian", self.theta.shape[0]*self.theta.shape[0])
        p_gradient_sym = cs.MX.sym("gradient", self.theta.shape[0])

        delta_theta = cs.MX.sym("delta_theta", self.theta.shape[0])
        theta = cs.MX.sym("delta_theta", self.theta.shape[0])

        lambda_reg = 1e-6

        qp = {
            "x": cs.vertcat(delta_theta),
            "p": cs.vertcat(theta, Hessian_sym, p_gradient_sym),
            "f": 0.5*delta_theta.T @ Hessian_sym.reshape((self.theta.shape[0], self.theta.shape[0])) @ delta_theta +  
                p_gradient_sym.T @ (delta_theta) + lambda_reg/2 * delta_theta.T @ delta_theta, 
            "g": theta + delta_theta,
        }

        opts = {
            "error_on_fail": False,
            "print_time": False,
            "verbose": False,
            "max_io": False,
            "osqp": {
                "eps_abs": 1e-9,
                "eps_rel": 1e-9,
                "max_iter": 10000,
                "eps_prim_inf": 1e-9,
                "eps_dual_inf": 1e-9,
                "polish": True,
                "scaling": 100,
                "verbose": False,
            },
        }

        return cs.qpsol('solver','osqp', qp, opts)
    

####################### RL #######################


class RLclass:

        def __init__(
        self,
        params_innit,
        seed,
        alpha,
        gamma,
        decay_rate,
        layers_list,
        noise_scalingfactor,
        noise_variance,
        patience_threshold,
        lr_decay_factor,
        horizon,
        positions,
        radii,
        modes,
        mode_params,
        slack_penalty
        ):
            # Store random seed for reproducibility
            self.seed = seed

            # Create the environment
            self.env = env()
            
            # Penalty in RL stagecost on slacks
            self.slack_penalty = slack_penalty

            # Initialize MPC and obstacle‐motion classes 
            self.mpc = MPC(layers_list, horizon, positions, radii, self.slack_penalty)
            self.obst_motion = ObstacleMotion(positions, modes, mode_params)
            
            # Parameters of experiments and states
            self.ns = self.mpc.ns
            self.na = self.mpc.na
            self.horizon = self.mpc.horizon
            self.params_innit = params_innit

            # Learning‐rate for parameter updates
            self.alpha = alpha
            
            # Build state bounds repeated over the horizon
            #np.tile takes an array and “tiles” (i.e. repeats) it to fill a larger array.
            self.X_lower_bound = -np.tile(CONSTRAINTS_X, self.horizon)
            self.X_upper_bound = np.tile(CONSTRAINTS_X, self.horizon)
            
            # Equality‐constraint bounds (Ax+Bu==0) - all zeros
            self.state_const_lbg = np.zeros(1*self.ns * (self.horizon))
            self.state_const_ubg = np.zeros(1*self.ns  * (self.horizon))

            # CBF safety constraints: h(x_{k+1})-h(x_k)+alpha*h(x_k) + s >= 0 → we invert so g =< 0
            # means the cbf constraint is bounded between -inf and zero --> for the g =< 0
            self.cbf_const_lbg = -np.inf * np.ones(self.mpc.rnn.obst.obstacle_num*(self.horizon))
            self.cbf_const_ubg = np.zeros(self.mpc.rnn.obst.obstacle_num*(self.horizon))
            
            
            # Discount factor for TD updates
            self.gamma = gamma

            # RNG for adding exploration noise
            self.np_random = np.random.default_rng(seed=self.seed)
            self.noise_scalingfactor = noise_scalingfactor
            self.noise_variance = noise_variance
            
            # Create CasADi mpc solver instances once for reuse
            self.solver_inst = self.mpc.MPC_solver()  #deterministic MPC solver
            self.solver_inst_random =self.mpc.MPC_solver_rand() # noisy MPC (the MPC with exploration noise)
            
            # Symbolic function to get the gradient of the MPC Lagrangian
            self.qlagrange_fn_jacob, _ = self.mpc.generate_symbolic_mpcq_lagrange()

            # Create CasADi qp solver instance once for reuse
            self.qp_solver = self.mpc.qp_solver_fn() # QP for constrained parameter updates

            # Learning‐rate scheduling
            self.decay_rate      = decay_rate
            self.patience_threshold = patience_threshold
            self.lr_decay_factor   = lr_decay_factor
            self.best_stage_cost   = np.inf
            self.best_params       = params_innit.copy()
            self.current_patience = 0

            
            # ADAM
            # theta_vector_num = cs.vertcat(cs.diag(self.params_innit["P"]), self.params_innit["rnn_params"])
            # theta_vector_num = cs.vertcat(cs.diag(self.params_innit["P"]), self.params_innit["rnn_params"])
            theta_vector_num = cs.vertcat(self.params_innit["rnn_params"])
            self.exp_avg = np.zeros(theta_vector_num.shape[0])
            self.exp_avg_sq = np.zeros(theta_vector_num.shape[0])
            self.adam_iter = 1
            
            # hidden state 1 function 
            self.get_hidden_func = self.mpc.rnn.make_rnn_step()
            
            self.flat_input_fn = self.mpc.make_flat_input_fn()
            
            
            # Warmstart variables storage
            self.x_prev_VMPC        = cs.DM()  
            self.lam_x_prev_VMPC    = cs.DM()  
            self.lam_g_prev_VMPC    = cs.DM()  
            self.x_prev_QMPC        = cs.DM()  
            self.lam_x_prev_QMPC    = cs.DM()  
            self.lam_g_prev_QMPC    = cs.DM()  
            self.x_prev_VMPCrandom  = cs.DM()  
            self.lam_x_prev_VMPCrandom = cs.DM()  
            self.lam_g_prev_VMPCrandom = cs.DM()
            
        
        def save_figures(self, figures, experiment_folder, save_in_subfolder=False):
            # Choose to save or not save figure
            save_choice = True  
            
            if save_choice:
                # Decide which subfolder (if any) to use
                if save_in_subfolder == "Learning":
                    target_folder = os.path.join(experiment_folder, "learning_process")
                elif save_in_subfolder == "Evaluation":
                    target_folder = os.path.join(experiment_folder, "evaluation")
                else:
                    # No subfolder specified: save directly in experiment_folder
                    target_folder = experiment_folder

                # Create the directory if it doesn’t exist
                os.makedirs(target_folder, exist_ok=True)
                
                # Loop through (figure, filename) pairs
                for fig, filename in figures:
                    file_path = os.path.join(target_folder, filename)
                    # Save the matplotlib figure 
                    fig.savefig(file_path)
                    plt.close(fig)
                    print(f"Figure saved as: {file_path}")
            else:
                print("Figure not saved")
                
        def make_system_obstacle_animation(self,
            states: np.ndarray,
            obs_positions: np.ndarray,
            radii: list,
            constraints_x: float,
            out_path: str,
            ):
            """
            
            Args:
                states        : (T,4) array of system [x,y,vx,vy]
                obs_positions : (T, m, 2) array of obstacle centers
                radii         : list of length m
                constraints_x : scalar for plotting window
                out_path      : path to save the .gif
                """ 
                
            # Number of time steps T and number of obstacles m
            T, m = obs_positions.shape[:2]
            # Extract only the (x,y) positions of the system over time
            system_xy = states[:, :2]

            fig, ax = plt.subplots()
            ax.set_aspect("equal", "box")
            ax.grid(True)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(r"System + Moving Obstacles")
            # Set fixed window of gif based on constraints_x
            span = constraints_x
            ax.set_xlim(-1.1*span, +0.1*span)
            ax.set_ylim(-1.1*span, +0.1*span)

            # Prepare the system path (line) and current position (dot)
            line, = ax.plot([], [], "o-", lw=2, label=r"system path")
            dot,  = ax.plot([], [], "ro", ms=6,    label=r"system")

            # Get a set of distinct colors for obstacles
            cmap   = plt.get_cmap("tab10")
            colors = cmap.colors  # this is a tuple-list of length 10

            # Ensure output directory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # one circle per obstacle
            circles = []
            for i, r in enumerate(radii):
                c = plt.Circle(
                    (0, 0), r,
                    fill=False,
                    color=colors[i % len(colors)],
                    lw=2,
                    label=f"obstacle {i+1}"
                )
                ax.add_patch(c)
                circles.append(c)

            ax.legend(loc="upper right")

            def init():
                """Initialize the plot elements for animation."""
                line.set_data([], [])
                dot.set_data([], [])
                for c in circles:
                    c.center = (0, 0)
                return [line, dot] + circles

            def update(k):
                """Update the plot elements for frame k."""
                # system
                xk, yk = system_xy[k]
                line.set_data(system_xy[:k+1, 0], system_xy[:k+1, 1])
                dot.set_data(xk, yk)
                # obstacles
                for i, c in enumerate(circles):
                    cx, cy = obs_positions[k, i]
                    c.center = (cx, cy)
                return [line, dot] + circles
            
            # Create the animation: T frames, with init and update functions
            ani = animation.FuncAnimation(
                fig, update, frames=T, init_func=init,
                blit=True, interval=100
            )
            ani.save(out_path, writer="pillow", fps=3, dpi=90)
            plt.close(fig)

        def plot_B_update(self, B_update_history, experiment_folder):
            
            """"
            B_update_history  is a history if update vectors for the RL parameters
            """
            
            B_update = np.asarray(B_update_history)
            B_update = B_update.squeeze(-1)

            # Build labels for the first four diagonal P elements
            labels = [f"P[{i},{i}]" for i in range(4)]
            print(f"labels: {labels}")

            # The remaining columns correspond to RNN parameter updates
            nn_B_update = B_update[:, 4:]
            # Compute mean absolute update magnitude across RNN parameters for each iteration
            # take mean across rows (7,205) --> (7,)
            mean_mag = np.mean(np.abs(nn_B_update), axis=1)
            
            # #legend helper function
            # def safe_legend(loc="best", **kwargs):
            #     handles, labls = plt.gca().get_legend_handles_labels()
            #     if labls:
            #         plt.legend(loc=loc, **kwargs)

            # Plot updates for P parameters
            fig_p = plt.figure()
            for idx, lbl in enumerate(labels):
                plt.plot(B_update[:, idx], "o-", label=lbl)
            plt.xlabel("Update iteration")
            plt.ylabel("B_update")
            plt.title("P parameter B_update over training")
            plt.legend()
            plt.grid(True)
            # safe_legend(loc="upper right", fontsize="small")
            plt.tight_layout()
            self.save_figures([(fig_p, "P_B_update_over_time")], experiment_folder)
            plt.close(fig_p)

            # Plot the RNN mean
            fig_nn = plt.figure()
            plt.plot(mean_mag, "o-", label="mean abs(NN_B_update)")
            plt.xlabel("Update iteration")
            plt.ylabel("Mean absolute B_update")
            plt.title("RNN mean across RNN params B_update magnitude over training")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            self.save_figures([(fig_nn, "NN_mean_B_update_over_time")], experiment_folder)
            plt.close(fig_nn)
        
        def ADAM(self, iteration, gradient, exp_avg, exp_avg_sq,
            learning_rate, beta1, beta2, eps = 1e-8): 
            """
            Computes the update's change according to Adam algorithm.
            
            Args:
                iteration (int): current iteration number (aka how many times ADAM has been called).
                gradient (array-like): raw gradient vector (delta theta).
                exp_avg (np.ndarray): running first moment estimate (EMA of the gradient; tracks direction).
                exp_avg_sq (np.ndarray): running second moment estimate (EMA of squared gradient; tracks magnitude).
                learning_rate (float or array-like): base step size.
                beta1 (float): decay rate for the first moment (e.g. 0.9).
                beta2 (float): decay rate for the second moment (e.g. 0.999).
                eps (float): small constant to avoid division by zero.

            Returns:
                dtheta (np.ndarray): the computed parameter increment used for update (delta theta)
                exp_avg (np.ndarray): updated first moment estimate.
                exp_avg_sq (np.ndarray): updated second moment estimate.
            """
            gradient = np.asarray(gradient).flatten()

            exp_avg = beta1 * exp_avg + (1 - beta1) * gradient
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * np.square(gradient)

            bias_correction1 = 1 - beta1**iteration                
            bias_correction2 = 1 - beta2**iteration

            step_size = learning_rate / bias_correction1
            bias_correction2_sqrt = np.sqrt(bias_correction2)

            denom = np.sqrt(exp_avg_sq) / (bias_correction2_sqrt + eps)
            
            dtheta = -step_size * (exp_avg / denom)
            return dtheta, exp_avg, exp_avg_sq

        def update_learning_rate(self, current_stage_cost, params):
            """
            Update the learning rate based on the current stage cost metric.
            
            If the stage cost improves (goes down), reset patience and store the best params.
            If it doesn’t improve for `patience_threshold` calls, decay the learning rate,
            reset patience, and roll back to the best‐seen parameters.
            """
            if current_stage_cost < self.best_stage_cost:
                self.best_params = params.copy() 
                self.best_stage_cost = current_stage_cost
                self.current_patience = 0
            else:
                self.current_patience += 1

            if self.current_patience >= self.patience_threshold:
                old_alpha = self.alpha
                self.alpha *= self.lr_decay_factor  # decay 
                print(f"Learning rate decreased from {old_alpha} to {self.alpha} due to no stage cost improvement.")
                self.current_patience = 0  # reset 
                params = self.best_params  # revert to best params

            return params

        def noise_scale_by_distance(self, x, y, max_radius=3):
            
            
            """
            Compute a scaling factor for exploration noise based on distance from the origin. 
            Close to the origin, noise is scaled down; at max_radius, it is 1.

            Args:
                x (float): current x positon of the system.
                y (float): current y position of the system.
                max_radius (float): distance beyond which noise scaling caps at 1.

            Returns:
                float: a factor in [0, 1] by which to multiply noise.
            """
            dist = np.sqrt(x**2 + y**2)
            if dist >= max_radius:
                return 1
            else:
                return (dist / max_radius)

        def cholesky_added_multiple_identity(self,
            A, beta: float = 1e-3, maxiter: int = 1000
        ):
            r"""Lower Cholesky factorization with added multiple of the identity matrix to ensure
            positive-definitiveness from Algorithm 3.3 in :cite:`nocedal_numerical_2006`.

            The basic idea is to add a multiple of the identity matrix to the original matrix
            unitl the factorization is successful, i.e., find :math:`\tau \ge 0` such that
            :math:`L L^\top = A + \tau I` is successful.

            Parameters
            ----------
            A : array of double
                The 2D matrix to compute the cholesky factorization of.
            beta : float, optional
                Initial tolerance of the algorithm, by default ``1e-3``.
            maxiter : int, optional
                Maximum iterations of the algorithm, by default ``1000``.

            Returns
            -------
            array of double
                The lower cholesky factorization of the modified ``A`` (with the addition of
                identity matrices to ensure that it is positive-definite).

            Raises
            ------
            ValueError
                If the factorization is unsuccessful for the maximum number of iterations.
            """
            a_min = np.diag(A).min()
            tau = 0 if a_min > 0 else -a_min + beta
            identity = np.eye(A.shape[0])
            for _ in range(maxiter):
                try:
                    return np.linalg.cholesky(A + tau * identity)
                except np.linalg.LinAlgError:
                    tau = max(1.05 * tau, beta)
            print("Warning: Cholesky decomposition failed after maximum iterations; returning zero matrix.")
            return np.zeros((A.shape[0], A.shape[0]))

        def V_MPC(self, params, x, xpred_list, ypred_list, hidden_in):
            """
            Solve the value-function MPC problem for the current state.

            Args:
                params (dict):  
                    Dictionary of system and RNN parameters
                x (ns,):  
                    Current state of the system.
                xpred_list (m*(horizon+1),):  
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):  
                    Predicted obstacle y-positions over the horizon.
                hidden_in:  
                    Current hidden-state vectors for the RNN layers.

            Returns:
                u_opt (na,):  
                    The first optimal control action.
                V_val (solution["f"]):  
                    The optimal value function V(x).
                hidden_t1 :  
                    Updated hidden states after one RNN forward pass.
            """
            # bounds

            # input bounded between 1 and -1
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            # state constraints (first state is bounded to be x0), omega cannot be 0
            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound,  
                                  np.zeros(self.mpc.rnn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.array(x).flatten(), self.X_upper_bound, U_upper_bound, 
                                  np.inf*np.ones(self.mpc.rnn.obst.obstacle_num *self.horizon)])

            #lower and upper bound for state and cbf constraints 
            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten to put it into the solver 
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], 
                                                       P_diag, Q_flat, R_flat,  params["rnn_params"], 
                                                       xpred_list, ypred_list, *hidden_in),
                x0    = self.x_prev_VMPC,
                lam_x0 = self.lam_x_prev_VMPC, # warm‐start multipliers on x‐bounds
                lam_g0 = self.lam_g_prev_VMPC, # warm‐start multipliers on g
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )

            #extract first optimal control action to apply (MPC)
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]
            
            # calculate new hidden state of the RNN for V_MPC
            X = cs.reshape(solution["x"][:self.ns * (self.horizon+1)], self.ns, self.horizon + 1)
            flat_input = self.flat_input_fn(X, xpred_list, ypred_list)
            x_t0 = flat_input[:1*self.ns+self.mpc.rnn.obst.obstacle_num]
            params_rnn = self.mpc.rnn.unpack_flat_parameters(params["rnn_params"])
            *hidden_t1, _ = self.get_hidden_func(*hidden_in, x_t0, *params_rnn)
            
            # warmstart variables for next iteration
            self.x_prev_VMPC     = solution["x"]
            self.lam_x_prev_VMPC = solution["lam_x"]
            self.lam_g_prev_VMPC = solution["lam_g"]
            
            # remember the slack variables for stage cost computation (in the evaluation stage cost)
            self.S_VMPC = solution["x"][self.na * (self.horizon) + self.ns * (self.horizon+1):]

            return u_opt, solution["f"], hidden_t1
        
        def V_MPC_rand(self, params, x, rand, xpred_list, ypred_list, hidden_in):
            """
            Solve the value‐function MPC problem with injected randomness.

            This is identical to V_MPC, but includes a random noise term in the optimization
            to encourage exploration.

            Args:
                params (dict):
                    Dictionary of system and RNN parameters:
                x (ns,):
                    Current system state vector.
                rand (na,1):
                    Random noise vector added to first control action in MPC objective
                xpred_list (m*(horizon+1),):
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):
                    Predicted obstacle y-positions over the horizon.
                hidden_in (list of MX):
                    Current RNN hidden-state from previous time step.

            Returns:
                u_opt (na,):
                    The first optimal control action (with randomness).
                hidden_t1 (list of MX):
                    Updated RNN hidden-state 
            """
            
            # bounds
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound,  np.zeros(self.mpc.rnn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.array(x).flatten(),self.X_upper_bound, U_upper_bound,  np.inf*np.ones(self.mpc.rnn.obst.obstacle_num *self.horizon)])
            

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])#cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)


            solution = self.solver_inst_random(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], 
                                                              P_diag, Q_flat, R_flat, params["rnn_params"], 
                                                              rand, xpred_list, ypred_list, *hidden_in),
                x0    = self.x_prev_VMPCrandom,
                lam_x0 = self.lam_x_prev_VMPCrandom,
                lam_g0 = self.lam_g_prev_VMPCrandom,
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )
            #extract first optimal control action to apply (MPC)
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]
            
            # calculate new hidden state of the RNN for V_MPC_rand
            X = cs.reshape(solution["x"][:self.ns * (self.horizon+1)], self.ns, self.horizon + 1)
            flat_input = self.flat_input_fn(X, xpred_list, ypred_list)
            x_t0 = flat_input[:1*self.ns+self.mpc.rnn.obst.obstacle_num]
            params_rnn = self.mpc.rnn.unpack_flat_parameters(params["rnn_params"])  
            *hidden_t1, _ = self.get_hidden_func(*hidden_in, x_t0, *params_rnn)
            
            # warmstart variables for next iteration
            self.x_prev_VMPCrandom = solution["x"]
            self.lam_x_prev_VMPCrandom = solution["lam_x"]
            self.lam_g_prev_VMPCrandom = solution["lam_g"]
            
            # remember the slack variables for stage cost computation (in the RL stage cost)
            self.S_VMPC_rand = solution["x"][self.na * (self.horizon) + self.ns * (self.horizon+1):]

            return u_opt, hidden_t1

        def Q_MPC(self, params, action, x, xpred_list, ypred_list, hidden_in):
            
            """"
            
            Solve the Q-value MPC problem for current state and current action.
            
            Similar to V_MPC, but includes the action in the optimization and computes the Q-value.
            
            Args:
                params (dict):
                    Dictionary of system and RNN parameters.
                action (na,):
                    Current control action vector.
                x (ns,):
                    Current state of the system.
                xpred_list (m*(horizon+1),):
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):
                    Predicted obstacle y-positions over the horizon.
                hidden_in (list of MX):
                    Current hidden-state vectors for the RNN layers.
            Returns:
                x_opt (ns*(horizon+1),):
                    Optimal state trajectory over the horizon.
                Q_val (solution["f"]):
                    Optimal Q-value for the current state and action.
                lagrange_mult_g (solution["lam_g"]):
                    Lagrange multipliers for the constraints.
                lam_lbx (solution["lam_x"]):
                    Lagrange multipliers for the lower bounds on x.
                lam_ubx (solution["lam_x"]):
                    Lagrange multipliers for the upper bounds on x.
                lam_p (solution["lam_p"]):
                    Lagrange multipliers for the parameters.
                hidden_t1 (list of MX):
                    Updated hidden states after one RNN forward pass.
                    """

            # Build input‐action bounds (note horizon−1 controls remain free after plugging in `action`)
            U_lower_bound = -np.ones(self.na * (self.horizon-1))
            U_upper_bound = np.ones(self.na * (self.horizon-1))

            #Assemble full lbx/ubx: [ x0; X(1…H); action; remaining U; slack ]
            lbx = np.concatenate([np.asarray(x).flatten(), self.X_lower_bound, np.asarray(action).flatten(), U_lower_bound,  
                                  np.zeros(self.mpc.rnn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.asarray(x).flatten(), self.X_upper_bound, np.asarray(action).flatten(), U_upper_bound, 
                                  np.inf*np.ones(self.mpc.rnn.obst.obstacle_num *self.horizon)])

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])#cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, 
                                                       Q_flat, R_flat, params["rnn_params"], 
                                                       xpred_list, ypred_list, *hidden_in),
                x0    = self.x_prev_QMPC,
                lam_x0 = self.lam_x_prev_QMPC,
                lam_g0 = self.lam_g_prev_QMPC,
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )
           
            # Extract lagrange multipliers needed for the lagrangian:
            lagrange_mult_g = solution["lam_g"] 
            lam_lbx = -cs.fmin(solution["lam_x"], 0)
            lam_ubx = cs.fmax(solution["lam_x"], 0)
            lam_p = solution["lam_p"]
            
            # calculate new hidden state of the RNN for Q_MPC
            X = cs.reshape(solution["x"][:self.ns * (self.horizon+1)], self.ns, self.horizon + 1)
            flat_input = self.flat_input_fn(X, xpred_list, ypred_list)
            x_t0 = flat_input[:1*self.ns+self.mpc.rnn.obst.obstacle_num]
            params_rnn = self.mpc.rnn.unpack_flat_parameters(params["rnn_params"])  
            *hidden_t1, _ = self.get_hidden_func(*hidden_in, x_t0, *params_rnn)
            
            # warmstart variables for next iteration
            self.x_prev_QMPC = solution["x"]
            self.lam_x_prev_QMPC = solution["lam_x"]
            self.lam_g_prev_QMPC = solution["lam_g"]
            
            return solution["x"], solution["f"], lagrange_mult_g, lam_lbx, lam_ubx, lam_p, hidden_t1
            
        def stage_cost(self, action, state, S):
            """
            Computes the stage cost : L(s,a).
            
            Args:
                action: (na,):
                    Control action vector.
                state: (ns,):
                    Current state vector of the system
                S: (m*(horizon+1),):
                    Slack variables for the MPC problem, used in the stage cost.
                    Slacks that were used for relaxing CBF constraints in the MPC problem.
            
            Returns:
                float:
                    The computed stage cost value.
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 10, 10])
            Rstage = np.diag([1, 1])
            
            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action + np.sum(self.slack_penalty *S)
            )
        
        def evaluation_step(self, params, experiment_folder, episode_duration):
            """
            Run an evaluation episode using the current parameters and plot results.
            Args:
                params (dict):
                    Dictionary of system and RNN parameters.
                experiment_folder (str):
                    Path to the experiment folder where results will be saved.
                episode_duration (int):
                    Number of steps in the evaluation episode.
            """
            
            state, _ = self.env.reset(seed=self.seed, options={})
            self.obst_motion.reset()
            
            states_eval = [state]
            actions_eval = []
            stage_cost_eval = []
            
            xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
            
            hidden_in_VMPC = [cs.DM.zeros(self.mpc.rnn.layers_list[i+1], 1) 
                    for i in range(len(self.mpc.rnn.layers_list)-2)
                    ]
            
            obs_positions = [self.obst_motion.current_positions()]
            
            self.x_prev_VMPC        = cs.DM()  
            self.lam_x_prev_VMPC    = cs.DM()  
            self.lam_g_prev_VMPC    = cs.DM()  

            for i in range(episode_duration):
                action, _, hidden_in_VMPC = self.V_MPC(params=params, x=state, xpred_list=xpred_list, ypred_list=ypred_list, hidden_in=hidden_in_VMPC)

                statsv = self.solver_inst.stats()
                if statsv["success"] == False:
                    print("V_MPC NOT SUCCEEDED in EVALUATION")
                
                action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
                stage_cost_eval.append(self.stage_cost(action, state, self.S_VMPC))
                
                # print(f"evaluation step {i}, action: {action}, slack: {np.sum(5e4*self.S_VMPC)}")
                state, _, done, _, _ = self.env.step(action)
                states_eval.append(state)
                actions_eval.append(action)

                _ = self.obst_motion.step()
                xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)

                obs_positions.append(self.obst_motion.current_positions())

            states_eval = np.array(states_eval)
            actions_eval = np.array(actions_eval)
            stage_cost_eval = np.array(stage_cost_eval)
            stage_cost_eval = stage_cost_eval.reshape(-1)
            obs_positions = np.array(obs_positions) 
            
            sum_stage_cost = np.sum(stage_cost_eval)
            print(f"Stage Cost: {sum_stage_cost}")

            figstates=plt.figure()
            plt.plot(
                states_eval[:, 0], states_eval[:, 1],
                "o-"
            )

            # Plot the obstacle
            for (cx, cy), r in zip(self.mpc.rnn.obst.positions, self.mpc.rnn.obst.radii):
                        circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
                        plt.gca().add_patch(circle)
            plt.gca().add_patch(circle)
            plt.xlim([-CONSTRAINTS_X[0], 0])
            plt.ylim([-CONSTRAINTS_X[1], 0])

            # Set labels and title
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.title(r"Trajectories")
            plt.legend()
            plt.axis("equal")
            plt.grid()
            self.save_figures([(figstates,
            f"states_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder, "Evaluation")

            figactions=plt.figure()
            plt.plot(actions_eval[:, 0], "o-", label=r"Action 1")
            plt.plot(actions_eval[:, 1], "o-", label=r"Action 2")
            plt.xlabel(r"Iteration $k$")
            plt.ylabel(r"Action")
            plt.title(r"Actions")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            self.save_figures([(figactions,
            f"actions_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder,"Evaluation")


            figstagecost=plt.figure()
            plt.plot(stage_cost_eval, "o-")
            plt.xlabel(r"Iteration $k$")
            plt.ylabel(r"Cost")
            plt.title(r"Stage Cost")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            self.save_figures([(figstagecost,
            f"stagecost_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder, "Evaluation")
            
            figsvelocity=plt.figure()
            plt.plot(states_eval[:, 2], "o-", label=r"Velocity x")
            plt.plot(states_eval[:, 3], "o-", label=r"Velocity y")    
            plt.xlabel(r"Iteration $k$")
            plt.ylabel(r"Velocity Value")
            plt.title(r"Velocity Plot")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            self.save_figures([(figsvelocity,
            f"velocity_MPCeval_{self.eval_count}_S_{sum_stage_cost}.svg")],
            experiment_folder, "Evaluation")
            
            
            target_folder = os.path.join(experiment_folder, "evaluation")
            out_gif = os.path.join(target_folder, f"system_and_obstacle_{self.eval_count}_SC_{sum_stage_cost}.gif")
            self. make_system_obstacle_animation(
            states_eval,
            obs_positions,
            self.mpc.rnn.obst.radii,
            CONSTRAINTS_X[0],
            out_gif,
            )

            self.update_learning_rate(sum_stage_cost, params)

            self.eval_count += 1

            return 
        
        def parameter_updates(self, params, B_update_avg):
            
            #TODO: Finish commenting and cleaning this funciton and onwards

            """
            function responsible for carryin out parameter updates after each episode
            """
            # P_diag = cs.diag(params["P"])

            #vector of parameters which are differenitated with respect to
            # theta_vector_num = cs.vertcat(P_diag, params["rnn_params"])
            theta_vector_num = cs.vertcat(params["rnn_params"])

            identity = np.eye(theta_vector_num.shape[0])

            # print(f"before updates : {theta_vector_num}")

            # alpha_vec is resposible for the updates
            # alpha_vec = cs.vertcat(self.alpha*np.ones(3), self.alpha, self.alpha, self.alpha*np.ones(theta_vector_num.shape[0]-5)*1e-2)
            alpha_vec = cs.vertcat(self.alpha*np.ones(theta_vector_num.shape[0]))
            # alpha_vec = cs.vertcat(self.alpha*np.ones(theta_vector_num.shape[0]-2), self.alpha,self.alpha*1e-5)
            
            print(f"B_update_avg:{B_update_avg}")

            dtheta, self.exp_avg, self.exp_avg_sq = self.ADAM(self.adam_iter, B_update_avg, self.exp_avg, self.exp_avg_sq, alpha_vec, 0.9, 0.999)
            self.adam_iter += 1 

            print(f"dtheta: {dtheta}")

            # uncostrained update to compare to the qp update
            y = np.linalg.solve(identity, dtheta)
            theta_vector_num_toprint = theta_vector_num - (y)#self.alpha * y
            print(f"theta_vector_num no qp: {theta_vector_num_toprint}")

            # lbx = cs.vertcat(-np.inf*np.ones(5), -0.01*np.abs(theta_vector_num[5:]))
            # ubx = cs.vertcat(np.inf*np.ones(5), 0.01*np.abs(theta_vector_num[5:]))

            # lbx = cs.vertcat(-np.inf*np.ones(5), -0.0001*np.ones(theta_vector_num.shape[0]-5))
            # ubx = cs.vertcat(np.inf*np.ones(5), 0.0001*np.ones(theta_vector_num.shape[0]-5))
            
            # constrained update qp update
            solution = self.qp_solver(
                    p=cs.vertcat(theta_vector_num, identity.flatten(), dtheta),
                    # lbg=cs.vertcat(np.zeros(4), -np.inf*np.ones(theta_vector_num.shape[0]-4)),
                    # ubg = cs.vertcat(np.inf*np.ones(theta_vector_num.shape[0])),
                    lbg=cs.vertcat(-np.inf*np.ones(theta_vector_num.shape[0])),
                    ubg = cs.vertcat(np.inf*np.ones(theta_vector_num.shape[0])),
                    # ubx = ubx,
                    # lbx = lbx
                )
            stats = self.qp_solver.stats()


            if stats["success"] == False:
                print("QP NOT SUCCEEDED")
                theta_vector_num = theta_vector_num
            else:
                theta_vector_num = theta_vector_num + solution["x"]

            print(f"theta_vector_num: {theta_vector_num}")

            # P_diag_shape = self.ns*1
            # #constructing the diagonal posdef P matrix 
            # P_posdef = cs.diag(theta_vector_num[:P_diag_shape])

            # params["P"] = P_posdef
            # params["rnn_params"] = theta_vector_num[P_diag_shape:]       
            params["rnn_params"] = theta_vector_num  

            return params
        

        def rl_trainingloop(self, episode_duration, num_episodes, replay_buffer, episode_updatefreq, experiment_folder):
    
            #to store for plotting
            params_history_P = [self.params_innit["P"]]

            #for the for loop
            params = self.params_innit
            
            x, _ = self.env.reset(seed=self.seed, options={})
            # reset obstacle motion

            stage_cost_history = []
            sum_stage_cost_history = []
            TD_history = []
            TD_temp = []
            TD_episode = []
            B_update_history = []
            grad_temp = []
            obs_positions = [self.obst_motion.current_positions()]

      
            B_update_buffer = deque(maxlen=replay_buffer)
            

            states = [(x)]

            actions = []
            
            xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)

            #intialize
            k = 0
            self.error_happened = False
            self.eval_count = 1
            
            #TODO: Review how hidden states are initialized
            
            hidden_in_VMPCrand = [cs.DM.zeros(self.mpc.rnn.layers_list[i+1], 1) 
                 for i in range(len(self.mpc.rnn.layers_list)-2)
                 ]
            
            hidden_in_QMPC = [cs.DM.zeros(self.mpc.rnn.layers_list[i+1], 1) 
                 for i in range(len(self.mpc.rnn.layers_list)-2)
                 ]
            
            hidden_in_VMPC = [cs.DM.zeros(self.mpc.rnn.layers_list[i+1], 1) 
                 for i in range(len(self.mpc.rnn.layers_list)-2)
                 ]

            for i in range(1,episode_duration*num_episodes):  
                
                noise = self.noise_scalingfactor*self.noise_scale_by_distance(x[0],x[1])
                rand = noise * self.np_random.normal(loc=0, scale=self.noise_variance, size = (2,1))

                u, hidden_in_VMPCrand = self.V_MPC_rand(params=params, x=x, rand = rand, xpred_list=xpred_list, 
                                    ypred_list=ypred_list, hidden_in=hidden_in_VMPCrand)
                u = cs.fmin(cs.fmax(cs.DM(u), -1), 1)

                actions.append(u)

                statsvrand = self.solver_inst_random.stats()
                if statsvrand["success"] == False:
                    print("V_MPC_RANDOM NOT SUCCEEDED")
                    self.error_happened = True

     
                solution, Qcost, lagrange_mult_g, lam_lbx, lam_ubx, _, hidden_in_QMPC = self.Q_MPC(params=params, action=u, x=x, 
                                                                                   xpred_list=xpred_list, ypred_list=ypred_list,
                                                                                   hidden_in=hidden_in_QMPC)
     

                statsq = self.solver_inst.stats()
                if statsq["success"] == False:
                    print("Q_MPC NOT SUCCEEDED")
                    self.error_happened = True


                S = solution[self.na * (self.horizon) + self.ns * (self.horizon+1):]
                stage_cost = self.stage_cost(action=u,state=x, S=self.S_VMPC_rand)
                
                # enviroment update step
                x, _, done, _, _ = self.env.step(u)

                # append trajectory points for plotting
                states.append(x)

                #calculate V value

                # print(f"x_2: {x}")
                # print(f"params_3: {params}")

                _, Vcost, hidden_in_VMPC = self.V_MPC(params=params, x=x, xpred_list=xpred_list, ypred_list=ypred_list, hidden_in=hidden_in_VMPC)

                statsv = self.solver_inst.stats()
                if statsv["success"] == False:
                    print("V_MPC NOT SUCCEEDED")
                    self.error_happened = True

                # TD update
                TD = (stage_cost) + self.gamma*Vcost - Qcost

                U = solution[self.ns * (self.horizon+1):self.na * (self.horizon) + self.ns * (self.horizon+1)] 
                X = solution[:self.ns * (self.horizon+1)] 
            
                qlagrange_numeric_jacob=  self.qlagrange_fn_jacob(
                    params["A"],
                    params["B"],
                    params["b"],
                    params["Q"],
                    params["R"],
                    params["P"],
                    lam_lbx,
                    lam_ubx,
                    lagrange_mult_g,
                    X, U, S, 
                    params["rnn_params"],
                    xpred_list, 
                    ypred_list,
                    *hidden_in_QMPC 
                )

                # first order update
                B_update = -TD*qlagrange_numeric_jacob
                grad_temp.append(qlagrange_numeric_jacob)
                B_update_buffer.append(B_update)
                        
                stage_cost_history.append(stage_cost)
                if self.error_happened == False:
                    TD_episode.append(TD)
                    TD_temp.append(TD)
                else:
                    TD_temp.append(cs.DM(np.nan))
                    self.error_happened = False
                    
                _ = self.obst_motion.step()
        
                xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)

                obs_positions.append(self.obst_motion.current_positions())
                
                if (k == episode_duration):                     
                    # -1 because loop starts from 1
                    if (i-1) % (episode_duration*episode_updatefreq) == 0:
                        
                        # self.evaluation_step(S=S, params=params, experiment_folder=experiment_folder, episode_duration=episode_duration)
                        
                        print (f"updatedddddd")
                        B_update_avg = np.mean(B_update_buffer, 0)
                        B_update_history.append(B_update_avg)

                        params = self.parameter_updates(params = params, B_update_avg = B_update_avg)
                        
                        params_history_P.append(params["P"])

                        
                        self.noise_scalingfactor = self.noise_scalingfactor*(1-self.decay_rate)

                        print(f"noise scaling: {self.noise_scalingfactor}")


                    sum_stage_cost_history.append(np.sum(stage_cost_history))
                    TD_history.append(np.mean(TD_episode))

                    stage_cost_history = []
                    TD_episode = []


                    # plotting the trajectories under the noisy policies explored
                    current_episode = i // episode_duration
                    if (current_episode % 50) == 0:
                        states = np.array(states)
                        actions = np.asarray(actions)
                        TD_temp = np.asarray(TD_temp) 
                        obs_positions = np.array(obs_positions)

                        figstate=plt.figure()
                        plt.plot(
                            states[:, 0], states[:, 1],
                            "o-"
                        )

                        # Plot the obstacle
                        for (cx, cy), r in zip(self.mpc.rnn.obst.positions, self.mpc.rnn.obst.radii):
                            circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
                            plt.gca().add_patch(circle)
                        plt.xlim([-CONSTRAINTS_X[0], 0])
                        plt.ylim([-CONSTRAINTS_X[1], 0])

                        # Set labels and title
                        plt.xlabel(r"$x$")
                        plt.ylabel(r"$y$")
                        plt.title(r"Trajectories of states while policy is trained$")
                        plt.legend()
                        plt.axis("equal")
                        plt.grid()
                        self.save_figures([(figstate,
                            f"position_plotat_{i}")],
                            experiment_folder, "Learning")


                        figvelocity=plt.figure()
                        plt.plot(states[:, 2], "o-", label=r"Velocity x")
                        plt.plot(states[:, 3], "o-", label=r"Velocity y")    
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"Velocity Value")
                        plt.title(r"Velocity Plot")
                        plt.legend()
                        plt.grid()
                        plt.tight_layout()
                        self.save_figures([(figvelocity,
                            f"figvelocity{i}")],
                            experiment_folder, "Learning")

                        # Plot TD
                        indices = np.arange(len(TD_temp))
                        figtdtemp = plt.figure(figsize=(10, 5))
                        plt.scatter(indices,TD_temp, label=r"TD")
                        plt.yscale('log')
                        plt.title(r"TD Over Training (Log Scale) - Colored by Proximity")
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"TD")
                        plt.legend()
                        plt.grid(True)
                        self.save_figures([(figtdtemp,
                            f"TD_plotat_{i}")],
                            experiment_folder, "Learning")

                        figactions=plt.figure()
                        plt.plot(actions[:, 0], "o-", label=r"Action 1")
                        plt.plot(actions[:, 1], "o-", label=r"Action 2")
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"Action")
                        plt.title(r"Actions")
                        plt.legend()
                        plt.grid()
                        plt.tight_layout()
                        self.save_figures([(figactions,
                            f"action_plotat_{i}")],
                            experiment_folder, "Learning")

                        gradst = np.asarray(grad_temp)
                        gradst = gradst.squeeze(-1)


                        labels = [f"P[{i},{i}]" for i in range(4)]
                        nn_grads = gradst[:, 4:]
                        # take mean across rows (7,205) --> (7,)
                        mean_mag = np.mean(np.abs(nn_grads), axis=1)

                        P_figgrad = plt.figure()
                        
                        for idx, lbl in enumerate(labels):
                                plt.plot(gradst[:, idx], "o-", label=lbl)
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"P gradient")
                        plt.title(r"P parameter gradients over training")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        self.save_figures([(P_figgrad,
                            f"P_grad_plotat_{i}")],
                            experiment_folder, "Learning")



                        NN_figgrad = plt.figure()
                        plt.plot(mean_mag, "o-", label=r"mean abs(rnn grad)")

                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"rnn Mean absolute gradient")
                        plt.title(r"rnn mean acoss rnn params gradient magnitude over training")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        self.save_figures([(NN_figgrad,
                            f"NN_grad_plotat_{i}")],
                            experiment_folder, "Learning")
                        # plt.show()

                        target_folder = os.path.join(experiment_folder, "learning_process")
                        out_gif = os.path.join(target_folder, f"system_and_obstacle_{self.eval_count}_SC_{sum_stage_cost_history[-1]}.gif")
                        self. make_system_obstacle_animation(
                        states,
                        obs_positions,
                        self.mpc.rnn.obst.radii,
                        CONSTRAINTS_X[0],
                        out_gif,
                        )
                        
                        # #evaluation step
                        self.evaluation_step(params=params, experiment_folder=experiment_folder, episode_duration=episode_duration)
            
                    # self.evaluation_step(params=params, experiment_folder=experiment_folder, episode_duration=episode_duration)

                    # reset the environment and the obstacle motion
                    x, _ = self.env.reset(seed=self.seed, options={})
                    self.obst_motion.reset()
                    k=0
                    
                    # reset hidden states
                    hidden_in_VMPCrand = [cs.DM.zeros(self.mpc.rnn.layers_list[i+1], 1) 
                    for i in range(len(self.mpc.rnn.layers_list)-2)
                    ]
            
                    hidden_in_QMPC = [cs.DM.zeros(self.mpc.rnn.layers_list[i+1], 1) 
                        for i in range(len(self.mpc.rnn.layers_list)-2)
                        ]
                    
                    hidden_in_VMPC = [cs.DM.zeros(self.mpc.rnn.layers_list[i+1], 1) 
                        for i in range(len(self.mpc.rnn.layers_list)-2)
                        ]
                    
                    self.x_prev_VMPC        = cs.DM()  
                    self.lam_x_prev_VMPC    = cs.DM()  
                    self.lam_g_prev_VMPC    = cs.DM()  

                    self.x_prev_QMPC        = cs.DM()  
                    self.lam_x_prev_QMPC    = cs.DM()  
                    self.lam_g_prev_QMPC    = cs.DM()  

                    self.x_prev_VMPCrandom  = cs.DM()  
                    self.lam_x_prev_VMPCrandom = cs.DM()  
                    self.lam_g_prev_VMPCrandom = cs.DM()
            
                    states = [(x)]
                    TD_temp = []
                    actions = []
                    grad_temp = []
                    obs_positions = [self.obst_motion.current_positions()]
                    xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
                    
                    print("reset")
                    
                # k counter    
                k+=1
                
                #counter
                if i % 1000 == 0:
                    print(f"{i}/{episode_duration*num_episodes}")  

            #show trajectories
            # plt.show()
            # plt.close()

            params_history_P = np.asarray(params_history_P)

            TD_history = np.asarray(TD_history)
            sum_stage_cost_history = np.asarray(sum_stage_cost_history)

            self.plot_B_update(B_update_history, experiment_folder)
       

            figP = plt.figure(figsize=(10, 5))
            plt.plot(params_history_P[:, 0, 0], label=r"$P_{1,1}$")
            plt.plot(params_history_P[:, 1, 1], label=r"$P_{2,2}$")
            plt.plot(params_history_P[:, 2, 2], label=r"$P_{3,3}$")
            plt.plot(params_history_P[:, 3, 3], label=r"$P_{4,4}$")
            # plt.title("Parameter: P",        fontsize=24)
            plt.xlabel(r"Update Number",     fontsize=20)
            plt.ylabel(r"Value",             fontsize=20)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=16)
            plt.grid()
            plt.tight_layout()
            self.save_figures([(figP,
                   f"P.svg")],
                 experiment_folder)
            

            figstagecost = plt.figure()
            plt.plot(sum_stage_cost_history, 'o', label=r"Stage Cost")
            plt.yscale('log')
            # plt.title("Stage Cost Over Training (Log Scale)", fontsize=24)
            plt.xlabel(r"Episode Number",                    fontsize=20)
            plt.ylabel(r"Stage Cost",                        fontsize=20)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=16)
            plt.grid(True)
            plt.tight_layout()
            self.save_figures([(figstagecost,
                   f"stagecost.svg")],
                 experiment_folder)
            

            figtd = plt.figure()
            plt.plot(TD_history, 'o', label=r"TD")
            plt.yscale('log')
            plt.title(r"TD Over Training (Log Scale)")
            plt.xlabel(r"Episode Number")
            plt.ylabel(r"TD")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            self.save_figures([(figtd,
                   f"TD.svg")],
                 experiment_folder)

            cost = np.array(sum_stage_cost_history)
            episodes = np.arange(len(cost))

            # choose window size
            window = 100

            # wrap in a pandas Series so rolling().mean()/.std() produce full‐length outputs
            s = pd.Series(cost)
            running_mean = s.rolling(window, center=True, min_periods=1).mean().values
            running_std  = s.rolling(window, center=True, min_periods=1).std().values

            # plt
            figstagecost_nice = plt.figure(figsize=(10,5))
            ax = figstagecost_nice.add_subplot(1,1,1)

            # running mean
            ax.plot(episodes, running_mean, '-', linewidth=2, label=rf"Stage Cost mean ({window}-ep)")

            # ±1σ band
            ax.fill_between(episodes,
                            running_mean - running_std,
                            running_mean + running_std,
                            alpha=0.3,
                            label=rf"Stage Cost std ({window}-ep)")

            if np.any(cost > 0):
                ax.set_yscale('log')

            ax.set_xlabel(r"Episode Number", fontsize=20)
            ax.set_ylabel(r"Stage Cost",     fontsize=20)
            ax.tick_params(labelsize=12)
            ax.grid(True)
            ax.legend(fontsize=16)
            figstagecost_nice.tight_layout()
            self.save_figures([(figstagecost_nice,
                   f"stagecost_smoothed.svg")],
                 experiment_folder)

            return params
        
        