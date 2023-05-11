#!/usr/bin/env python3
from nw_task_init import curvature_saccade_task

class uniform_curvature_task(curvature_saccade_task):
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

        params['s'] = 0.05 + (trial%20)/20
        if trial<40:
            params['a'] = -20
        elif trial<80:
            params['a'] = 0
        else:
            params['a'] = 20

        if trial<20:
            params['b'] = 100
        elif trial<40:
            params['b'] = 140
        elif trial<60:
            params['b'] = 100
        elif trial<80:
            params['b'] = 140
        elif trial<100:
            params['b'] = 100
        else:
            params['b'] = 140

        params['o'] = params['s']*params['b'] - params['b']/2 + params['a']
        
        params['s_onset'] = (120+250)/2
        params['ab_onset'] = params['s_onset'] + (500+800)/2
        params['fix_offset'] = params['ab_onset'] + (120+250)/2
        
        time_for_choice = 500
        params['fix_onset'] = self.T - params['fix_offset'] - time_for_choice
        if params['fix_onset']<0:
            params['fix_onset'] = 0
        
        params['s_onset'] = params['s_onset'] + params['fix_onset']
        params['ab_onset'] = params['ab_onset'] + params['fix_onset']
        params['fix_offset'] = params['fix_offset'] + params['fix_onset']
        
        return params

