from nw_task_init import curvature_saccade_task
from psychrnn.backend.models.basic import Basic # Basic=vanilla RNN. Can also use LSTMs, etc.
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def train_model(model_id):
  model_name = 'model' + str(model_id)
  N_trainbatch = 200 # number of trials per training update

  task = RamTask_v0(N_batch = N_trainbatch)

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

  model = Basic(network_params)

  train_params = dict() # specify training parameters. default transfer function is ReLU (can change if desired).
  train_params['training_iters'] = 200000
  train_params['learning_rate'] = 0.0005

  # TRAIN
  losses, initialization_time, training_time = model.train(task, train_params)
  model.save('weights/' + model_name)
  model.destruct()

  plt.figure(figsize=(5,5))
  plt.plot(losses)
  plt.title('Loss during training')
  plt.ylabel('Minibatch loss')
  plt.xlabel('Batch number')
  plt.savefig('plots/' + model_name + '_trainingLoss', dpi=200)

if __name__ == '__main__':
  p = Pool(processes=cpu_count())
  p.map(train_model, range(cpu_count()))