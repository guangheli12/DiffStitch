import shutil 
import argparse 
import time 
import pickle 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import diffuser.utils as utils
import torch
from copy import deepcopy
import random 
import numpy as np
import gym
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
from scripts.buffer_utils import LoadSequenceDataset
from scripts.buffer_utils import OptimalBuffer
from scripts.mopo.utils_trans import RewardPredictingModel 
from scripts.buffer_utils import augment_trajectories_new
import importlib  

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"

def import_config(config_name):
    module_path = f"detail_configs.{config_name}"
    try:
        module = importlib.import_module(module_path)
        return module.Config    
    except ImportError:
        print(f"Error: Module '{config_name}' not found or has no 'Config' attribute.")
        return None

def apply_dict(fn, d, *args, **kwargs):
	return {
		k: fn(v, *args, **kwargs)
		for k, v in d.items()
	}

def cycle(dl):
	while True:
		for data in dl:
			yield data

def seconds_to_hours_minutes(seconds):
	hours, remainder = divmod(seconds, 3600)
	minutes, _ = divmod(remainder, 60)
	return hours, minutes

def get_bad_traj(dataset, num_optim): 
	from tqdm import tqdm 
	rec = [] 
	for i in tqdm(range(dataset.num_traj)): 
		info = dataset.get_full_info_traj(i)  
		discounted_return = 0 
		for j in range(info['horizon'] - 1, -1, -1): 
			discounted_return = discounted_return * 0.99 + info['rew'][j] 
			if info['horizon'] - j > 100: 
				rec.append([i, j, discounted_return])      
	rec = sorted(rec, key = lambda item: item[2])  
	dic_ret = {} 
	for i in range(30):
		dic_ret[str(i)] = [] 
	for i in tqdm(rec): 
		if i[2] < 0: 
			continue
		dic_ret[str(int(i[2] / 50))].append(i)   
	for i in range(30):
		print(f'{i} : ({i * 50} , {(i + 1) * 50}] = {len(dic_ret[str(i)])}')    
	return rec[-num_optim:], dic_ret 


def sample_traj_from_dict(rec): 
	import random 
	num = random.randint(0, len(rec) - 1) 
	return rec[num]   

def evaluate(**deps):
	from ml_logger import logger 
	Config = import_config(config_name = deps['config']) 

	Config._update(deps)
	Config.device = torch.device(Config.device) 
	BASE_DATA_PATH = Config.save_data_path 
	Config.save_data_path = os.path.join(
		BASE_DATA_PATH, 
		Config.dataset, 
		str(Config.number_optimum) + '_' + str(Config.top_k) + '_' + str(Config.dreamer_similarity) + '_' + str(Config.dynamics_deviate) + '_' + str(Config.stitch_L) + '_' + str(Config.stitch_R),
		'data' 
	)

	Config.save_img_dir = os.path.join(
		Config.save_img_dir, 
		Config.dataset, 
		str(Config.number_optimum) + '_' + str(Config.top_k) + '_' + str(Config.dreamer_similarity) + '_' + str(Config.dynamics_deviate) + '_' + str(Config.stitch_L) + '_' + str(Config.stitch_R)
	)

	save_config_path = os.path.join(
		BASE_DATA_PATH, 
		Config.dataset, 
		str(Config.number_optimum) + '_' + str(Config.top_k) + '_' + str(Config.dreamer_similarity) + '_' + str(Config.dynamics_deviate) + '_' + str(Config.stitch_L) + '_' + str(Config.stitch_R),
		'config' 
	)

	if os.path.exists(save_config_path):
		shutil.rmtree(save_config_path)
		print(f"folder '{save_config_path}' is flushed")
	logger.configure(save_config_path) 
	logger.log_params(Config=vars(Config))       

	dataset_config = utils.Config(
		'datasets.SequenceDataset',
		savepath='dataset_config.pkl',
		env=Config.dataset,
		horizon=Config.horizon,
		normalizer=Config.normalizer,
		preprocess_fns=Config.preprocess_fns,
		use_padding=Config.use_padding,
		max_path_length=Config.max_path_length,
		include_returns=Config.include_returns,
		returns_scale=Config.returns_scale,
	)

	render_config = utils.Config(
		Config.renderer,
		savepath='render_config.pkl',
		env=Config.dataset,
	)

	dataset = dataset_config()
	renderer = render_config()


	loadpath = os.path.join(Config.bucket, Config.prefix, 'checkpoint')
	print('\n\nloadpath = ', loadpath, end = '\n\n')  

	# load it from checkpoint
	if Config.save_checkpoints:
		loadpath = os.path.join(loadpath, f'state_1000000.pt')
	else: 
		loadpath = os.path.join(loadpath, 'state.pt')
	
	state_dict = torch.load(loadpath, map_location=Config.device)

	# Load configs
	torch.backends.cudnn.benchmark = True
	utils.set_seed(Config.seed)
	random.seed(Config.seed)  

	observation_dim = dataset.observation_dim
	action_dim = dataset.action_dim

	if Config.diffusion == 'models.GaussianInvDynDiffusion':
		transition_dim = observation_dim
	else:
		transition_dim = observation_dim + action_dim

	model_config = utils.Config(
		Config.model,
		savepath='model_config.pkl',
		horizon=Config.horizon,
		transition_dim=transition_dim,
		cond_dim=observation_dim,
		dim_mults=Config.dim_mults,
		dim=Config.dim,
		returns_condition=Config.returns_condition,
		device=Config.device,
	)

	diffusion_config = utils.Config(
		Config.diffusion,
		savepath='diffusion_config.pkl',
		horizon=Config.horizon,
		observation_dim=observation_dim,
		action_dim=action_dim,
		n_timesteps=Config.n_diffusion_steps,
		loss_type=Config.loss_type,
		clip_denoised=Config.clip_denoised,
		predict_epsilon=Config.predict_epsilon,
		hidden_dim=Config.hidden_dim,
		## loss weighting
		action_weight=Config.action_weight,
		loss_weights=Config.loss_weights,
		loss_discount=Config.loss_discount,
		returns_condition=Config.returns_condition,
		device=Config.device,
		condition_guidance_w=Config.condition_guidance_w,
	)

	trainer_config = utils.Config(
		utils.Trainer,
		savepath='trainer_config.pkl',
		train_batch_size=Config.batch_size,
		train_lr=Config.learning_rate,
		gradient_accumulate_every=Config.gradient_accumulate_every,
		ema_decay=Config.ema_decay,
		sample_freq=Config.sample_freq,
		save_freq=Config.save_freq,
		log_freq=Config.log_freq,
		label_freq=int(Config.n_train_steps // Config.n_saves),
		save_parallel=Config.save_parallel,
		bucket=Config.bucket,
		n_reference=Config.n_reference,
		train_device=Config.device,
	)

	model = model_config()
	diffusion = diffusion_config(model)
	trainer = trainer_config(diffusion, dataset, renderer)
	# logger.print(utils.report_parameters(model), color='green')
	trainer.step = state_dict['step']
	trainer.model.load_state_dict(state_dict['model'])
	trainer.ema_model.load_state_dict(state_dict['ema'])
	#----------------------------------augmentation-----------------------------# 
	env_dataset    = LoadSequenceDataset(env_name = Config.dataset)
	top_k_trajs, dic_ret = get_bad_traj(env_dataset, Config.top_k)   
	dynamics_model = RewardPredictingModel(
		device = Config.device, 
		env_name = Config.dataset, 
		load_path = Config.dynamic_model_path
	) 
	expert_buffer  = OptimalBuffer()
	for i in range(env_dataset.num_traj): 
		info = env_dataset.get_full_info_traj(i)  
		expert_buffer.insert_traj(info) 

	number_optimum = Config.number_optimum
	expert_buffer.info = sorted(expert_buffer.info, key = lambda item: item[0])[-number_optimum:] 
	print(f'Optimal Buffer Range : [{expert_buffer.info[0][0]},  {expert_buffer.info[-1][0]}]')

	
	augmented_infos = []
	cnt = 0 
	total_trans = 0 
	conc_start = time.time() 
	stitch_batch_size = Config.stitch_batch 

	total_number_traj = 0
	filtered_number_traj = 0 
	# iterate until generated number reaches limit 
	for number_it in range(300000):
		expert_infos = [] 
		for idx in range(0, stitch_batch_size): 
			traj_i = sample_traj_from_dict(top_k_trajs) 
			i = traj_i[0] 
			expert_info_i = env_dataset.get_full_info_traj(i) 

			l_1 = traj_i[1]
			r_1 = l_1 + 100

			# l_1 
			original_discounted_reward = 0
			for t in range(expert_info_i['horizon'] - 1, l_1 - 1, -1): 
				original_discounted_reward = original_discounted_reward * 0.99 + expert_info_i['rew'][t] 
			
			for _ in ['act', 'obs', 'rew']: 
				expert_info_i[_] = expert_info_i[_][l_1 : r_1]
			expert_info_i['original_discounted_reward'] = original_discounted_reward
			expert_infos.append(expert_info_i) 
			
		time_taken = time.time() - conc_start 

		print('\n\n') 
		print(GREEN + f'### Iterate Number : {number_it}' + RESET)  
		print('----------------------') 
		# time_hours, time_minutes = seconds_to_hours_minutes(time_taken)
		time_hours = int(time_taken) // 3600 
		time_minutes = int((time_taken - (time_hours * 3600))) // 60
		print(f'Time : {time_hours} h , {time_minutes} m')   
		print(f'Total Number is {total_trans}') 

		if total_number_traj > 0: 
			print(f'FILTER RATIO: {round(filtered_number_traj / total_number_traj * 100, 2)}%')  

		if number_it > 1: 
			print(f'ACCEPT RATIO: {cnt} / {number_it * stitch_batch_size} = {round(cnt / (number_it * stitch_batch_size) * 100, 2)}%') 
		if cnt > 1: 
			print(f'average of {round(time_taken / (cnt), 2)} per traj')
			print(f'speed: {round(total_trans / time_taken, 2)} data/s') 
			print(f'Time remaining: {round((Config.generate_limit - total_trans) / (total_trans / time_taken) / 3600, 2)} h')
		print('----------------------') 
		print('\n') 

		render_option = False 
		if number_it % Config.render_freq == 0: 
			render_option = Config.render_option 
		augmented_transitions, generated_transitions, filter_stat = augment_trajectories_new(
			expert_infos = expert_infos, 
			dataset = dataset, 
			renderer = renderer, 
			trainer = trainer, 
			dynamics_model = dynamics_model,
			action_dim = action_dim, 
			obs_dim = observation_dim, 
			save_dir = os.path.join(Config.save_img_dir, str(number_it)),  
			dream_len = Config.dream_len, 
			horizon = Config.horizon,     
			device = Config.device, 
			save_name = str(number_it), 
			render_option = render_option, 
			dynamics_deviate = Config.dynamics_deviate,  
			test_ret = Config.test_ret, 
			expert_buffer = expert_buffer, 
			sample_optim_batch = Config.sample_optim_batch,
			original_discounted_reward = original_discounted_reward, 
			dreamer_similarity = Config.dreamer_similarity, 
			stitch_L = Config.stitch_L, 
			stitch_R = Config.stitch_R, 
			stitch_batch_size = stitch_batch_size
		)   

		total_number_traj += filter_stat[0] 
		filtered_number_traj += filter_stat[1] 

		if len(augmented_transitions) == 0: 
			continue  
		
		
		augmented_infos += augmented_transitions

		total_trans += generated_transitions

		cnt += len(augmented_transitions)
		print(GREEN + '####' + RESET)
		print(f'This is the {cnt} traj, current number is {generated_transitions}') 
		print(f'total number is {total_trans}') 
		print(GREEN + '####' + RESET)

		file_path = Config.save_data_path
		if number_it % Config.save_aug_freq == 0:       
			os.makedirs(Config.save_data_path, exist_ok = True)         
			file_path = os.path.join(
				Config.save_data_path, 
				Config.dataset + '_' + str(int((number_it // Config.save_aug_freq) % 2)) + '.pkl'
			)    
			with open(file_path, "wb") as file:       
				pickle.dump(augmented_infos, file)        
		if total_trans >= Config.generate_limit:   
			break 
	sys.exit(0)  

if __name__ == '__main__': 
	parser = argparse.ArgumentParser('DiffStitch', add_help=False)  
	parser.add_argument('--config', type=str) 
	parser.add_argument('--device', default='cuda:0', type=str)
	args = vars(parser.parse_args())   
	evaluate(**args)  