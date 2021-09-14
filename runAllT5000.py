import tensorflow as tf
from psychrnn.backend.models.basic import Basic
from psychrnn.backend.gain.basic2 import Basic2
from psychrnn.backend.gain.loss import rt_mask_mse_06, rt_mask_mse_07, rt_mask_mse_08
from psychrnn.tasks.checker import Checkerboard2AFC

from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline


noise_list = [2, 2.2, 2.4]

for id in range(len(noise_list)):

	for round in range(10):
		experiment = 'interactive'
		name = 'basic'

		# setup task parameters
		dt = 10
		tau = 50
		T = 5000
		N_batch = 50
		N_rec = 100
		# initialize task
		task = Checkerboard2AFC(dt=dt, tau=tau, T=T, N_batch=N_batch)

		# setup model parameters
		network_params = task.get_task_params()
		network_params['name'] = name
		network_params['N_rec'] = N_rec
		###############################################
		## We are going to change this noise 
		###############################################
		network_params['rec_noise'] = noise_list[id]

		network_params["transfer_function"] = tf.nn.relu
		network_params["output_transfer_function"] = tf.nn.sigmoid

		network_params["loss_function"] = "rt_mask_mse"
		network_params["rt_mask_mse"] = rt_mask_mse_07

		try:
		    model.destruct()
		except:
		    pass
		# create a model
		model = Basic2(network_params)
		# setup training parameters
		trials = 50000
		train_params = {}
		train_params['training_iters'] = trials
		train_params['learning_rate'] = .001
		train_params['loss_epoch'] = 10
		train_params['save_training_weights_epoch'] = 1000 / N_batch
		train_params['training_weights_path'] = None
		###################################################
		## Change the name of the save path
		###################################################
		train_params['save_weights_path'] =  "./weights/basic2_T5000_recNoise" + str(network_params['rec_noise']) + "round" + str(round)

		# train model
		losses, initialTime, trainTime = model.train(task, train_params)
		#####################################################
		## save losses
		#####################################################
		loss_name = "./losses/basic2_T5000_recNoise" + str(network_params['rec_noise']) + "round" + str(round) + ".txt"
		print("Save losses to " + loss_name)
		with open(loss_name, 'w') as f:
		    for item in losses:
		        f.write("%s\n" % item)

		# generate test trials
		trials = 5000
		batches = int(np.ceil(trials / N_batch))

		rnn_state = np.zeros((trials, task.N_steps, model.N_rec))
		rnn_out = np.zeros((trials, task.N_steps, model.N_out))

		coherence = np.zeros(trials)
		green_side = np.zeros(trials)
		target_onset = np.zeros(trials)
		checker_onset = np.zeros(trials)

		decision = np.zeros(trials)
		rt = np.zeros(trials)

		# run test trials
		for b in tqdm(range(batches)):
		    x, y, m, params = task.get_trial_batch()
		    outputs, states, inputs = model.test(x)
		    
		    start_index = N_batch * b
		    end_index = N_batch * (b + 1)
		    rnn_state[start_index:end_index] = states
		    rnn_out[start_index:end_index] = outputs
		    
		    thr = np.where(outputs > 0.7)
		    
		    for i in range(N_batch):
		        index = start_index + i
		        
		        coherence[index] = params[i]["coherence"]
		        green_side[index] = params[i]["side"]
		        target_onset[index] = params[i]["target_onset"]
		        checker_onset[index] = params[i]["checker_onset"]
		        
		        thr_time = thr[1][thr[0]==i][0] if sum(thr[0]==i) > 0 else outputs.shape[1]
		        thr_unit = thr[2][thr[0]==i][0] if sum(thr[0]==i) > 0 else np.argmax(outputs[i, -1])
		        decision[index] = thr_unit
		        rt[index] = thr_time*task.dt - target_onset[index] - checker_onset[index]       


		# export summary to a csv file
		correct_side = np.array([gs if coh > 0 else abs(gs-1) for coh, gs in zip(coherence, green_side)])
		green_decision = np.array([int(dec == gs) for dec, gs in zip(decision, green_side)])
		checker_df = pd.DataFrame({'trial' : np.arange(trials),
		                           'coherence' : coherence,
		                           'coherence_bin' : np.round(coherence, 1),
		                           'green_side' : green_side,
		                           'correct_side' : correct_side,
		                           'target_onset' : target_onset,
		                           'checker_onset' : checker_onset,
		                           'decision' : decision,
		                           'green_decision' : green_decision,
		                           'decision_time' : rt,
		                           'correct_decision' : (decision == correct_side).astype(int)})
		checker_mean = checker_df.groupby('coherence_bin').mean().reset_index()

		#############################################
		## Save checker_df with a specific name 
		#############################################
		resultPath = "./resultData/summary_T5000_recNoise" + str(network_params['rec_noise']) + "round" + str(round) + ".csv"
		 
		checker_df.to_csv(resultPath)

		model.destruct()