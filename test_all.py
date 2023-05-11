import scipy as sp
from nw_task_init import curvature_saccade_task
from psychrnn.backend.simulation import BasicSimulator
from multiprocessing import Pool, cpu_count

def test_model(model_id):
  model_name = 'model' + str(model_id)
  N_testbatch = 1000 # number of trials to test

  task = curvature_saccade_task(N_batch = N_testbatch)
  
  network_params = task.get_task_params()
  network_params['name'] = model_name
  network_params['N_rec'] = 50 # number of hidden units
  network_params['rec_noise'] = 0.01 # recurrent noise

  # "biological constraints" (optional):
  network_params['autapses'] = True # whether or not hidden units can connect to themselves
  network_params['dale_ratio'] = None # dale ratio, e.g. 0.8 if 80% of units can only send excitatory projections and 20% can only send inhibitory projections

  # regularization (optional)
  network_params['L2_in'] = 0
  network_params['L2_rec'] = 0
  network_params['L2_out'] = 0
  network_params['L2_firing_rate'] = 0
  network_params['L1_in'] = 0
  network_params['L1_rec'] = 0
  network_params['L1_out'] = 0
  
  test_inputs, target_outputs, mask, trial_params = task.get_trial_batch()
  simulator = BasicSimulator(weights_path='weights/' + model_name + '.npz', params=network_params)

  outputs, state_vars = simulator.run_trials(test_inputs)
  
  # save data in matlab format
  # collect arrays in dictionary
  savedict = {
      'outputs' : outputs,
      'state_vars' : state_vars,
      'test_inputs' : test_inputs,
      'target_outputs' : target_outputs,
      'trial_params' : trial_params
  }
  sp.io.savemat('data/test_' + model_name + '.mat', savedict)

if __name__ == '__main__':
  p = Pool(processes=cpu_count())
  p.map(test_model, range(cpu_count()))