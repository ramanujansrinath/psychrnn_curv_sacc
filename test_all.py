import scipy as sp
from nw_task_init import curvature_saccade_task
from psychrnn.backend.simulation import BasicSimulator
from multiprocessing import Pool, cpu_count

def test_model(model_id):
  model_name = 'model' + str(model_id)
  N_testbatch = 1000 # number of trials to test
  task = RamTask_v0(N_batch=N_testbatch)
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