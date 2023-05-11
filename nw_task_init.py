#!/usr/bin/env python3

from psychrnn.tasks.task import Task
import numpy as np
rng = np.random.default_rng(134)

class curvature_saccade_task(Task):
    def __init__(self, dt=10, tau=20, T=1800, N_in=4, N_out=1, N_batch=100, in_noise=0.01, alpha=1):
        # ----------------------------------
        # Define network parameters
        # ----------------------------------
        super().__init__(N_in, N_out, dt, tau, T, N_batch)
        self.in_noise = in_noise
                
    def generate_trial_params(self, batch, trial):
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
        params = dict()
        params['s'] = rng.uniform(0, 1)
        params['a'] = rng.choice([-20, 0, 20])
        params['b'] = rng.choice([100, 140])
        params['o'] = params['s']*params['b'] - params['b']/2 + params['a']
        
        params['s_onset'] = rng.uniform(120, 250)
        params['ab_onset'] = params['s_onset'] + rng.uniform(500, 800)
        params['fix_offset'] = params['ab_onset']+rng.uniform(120, 250)
        
        time_for_choice = 500
        params['fix_onset'] = self.T - params['fix_offset'] - time_for_choice
        if params['fix_onset']<0:
            params['fix_onset'] = 0
        
        params['s_onset'] = params['s_onset'] + params['fix_onset']
        params['ab_onset'] = params['ab_onset'] + params['fix_onset']
        params['fix_offset'] = params['fix_offset'] + params['fix_onset']
        
        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool or float, shape=(N_out,))): True or 1 if the network should train to match the y_t, False or 0 if the network should ignore y_t when training. Can also be an arbitrary scaling value.
        """
        
        # ----------------------------------
        # Initialize with input noise
        # ----------------------------------

        x_t = np.sqrt(2 * self.alpha * self.in_noise**2) * rng.standard_normal(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out) # weighing all timepoints and outputs equally
        
        # ----------------------------------
        # Compute values
        # ----------------------------------
        
        if time > params['fix_onset'] and time < params['fix_offset']:
            x_t[3] += 1
        
        if time > params['s_onset']:
            x_t[0] += params['s']
        
        if time > params['ab_onset']:
            x_t[1] += params['a']
            x_t[2] += params['b']
        
        if time > params['fix_offset']:
            y_t[0] = params['o']
            # y_t[1] = params['s']
        
        return x_t, y_t, mask_t

