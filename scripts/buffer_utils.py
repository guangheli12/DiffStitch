import random 
import d4rl
import gym
import sys 
import numpy as np 
import pandas as pd 
import pickle
import tqdm
import torch 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffuser.utils.arrays import to_torch, to_np, to_device
from scripts.mopo.utils_trans import RewardPredictingModel

def cosine_similarity(x, y):
	x_norm = np.linalg.norm(x)
	y_norm = np.linalg.norm(y, axis=1)
	dot_product = np.dot(y, x)
	similarity = dot_product / (x_norm * y_norm)
	return similarity

def augment_trajectories_new(
	expert_infos, 
	dataset, 
	renderer, 
	trainer, 
	dynamics_model, 
	action_dim, 
	obs_dim,
	save_dir,
	dream_len,
	horizon,
	device,
	save_name,
	render_option,
	dynamics_deviate, 
	test_ret, 
	expert_buffer,
	sample_optim_batch,
	original_discounted_reward,
	dreamer_similarity,
	stitch_L,
	stitch_R,
	stitch_batch_size
):
	traj1 = expert_infos
	traj_1_obs = [ np.expand_dims(_['obs'], axis = 0) for _ in traj1]     # traj_1_obs.shape is (100, 11)

	traj_1_obs = np.concatenate(traj_1_obs, axis = 0)
	for i in range(0, traj_1_obs.shape[0]):
		traj_1_obs[i] = dataset.normalizer.normalize(np.expand_dims(traj_1_obs[i], axis = 0), 'observations')

	cond = np.ones(shape = (stitch_batch_size, horizon, 2 * obs_dim)) 
	cond[:, :, obs_dim:] = 0
	cond[:, :dream_len, :obs_dim] = 0
	cond[:, :dream_len, obs_dim:] = traj_1_obs[:, :dream_len, :] 
	conditions = torch.tensor(cond).to(device)
	returns = to_device(test_ret * torch.ones(stitch_batch_size, 1), device)

	samples = trainer.ema_model.conditional_sample(conditions, returns=returns) # shape is [1, 100, 11]
	np_samples = to_np(samples)       # (32, 100, 11) 

	cond_p = np.ones(shape = (stitch_batch_size, horizon, 2 * obs_dim)) 
	mark_succ = np.ones(shape = (stitch_batch_size, )) 
	positions = [] 

	for i in range(0, stitch_batch_size): 

		# sample a batch of optimal trajectories 
		info_batch, obs_matrix = expert_buffer.sample_batch_traj(sample_optim_batch, dataset)


		# see which trajectory has a similarity greater than threshold
		available_positions = []
		maxim = -1 
		for cur_pos in range(stitch_L, stitch_R): 
			cosin_sim = cosine_similarity(np_samples[i][cur_pos], obs_matrix) 
			maxim = max(maxim, np.max(cosin_sim))
			if np.max(cosin_sim) < dreamer_similarity: 
				continue 
			available_positions += list(np.where(cosin_sim >= dreamer_similarity)[0])
		available_positions = list(set(available_positions)) 

		if len(available_positions) == 0: 
			mark_succ[i] = 0 
			positions.append([0, 0]) 
			continue
			
		# print(len(available_positions))


		chosen_index = random.randint(0, len(available_positions) - 1)    
		expert_info_j = {
			'obs': info_batch[available_positions[chosen_index]][1]
		} 
		traj2 = expert_info_j
		traj_2_obs = traj2['obs']    
		traj_2_obs = np.expand_dims(traj_2_obs, axis = 0)
		traj_2_obs = dataset.normalizer.normalize(traj_2_obs, 'observations')

		best_pos = 0
		best_ans = -1
		for _ in range(stitch_L, stitch_R):
			dist_i = cosine_similarity(traj_2_obs[0][0], np.expand_dims(np_samples[i][_], axis = 0))
			if dist_i > best_ans: 
				best_ans = dist_i
				best_pos = _

		start_index = dream_len
		end_index   = best_pos - 1

		positions.append([start_index, end_index, chosen_index])  
		cond_p[i][:, obs_dim:] = 0
		cond_p[i][:start_index, :obs_dim] = 0
		cond_p[i][:start_index, obs_dim:] = traj_1_obs[i][:start_index]
		cond_p[i][end_index + 1:, :obs_dim] = 0
		cond_p[i][end_index + 1:, obs_dim:] = traj_2_obs[0][:horizon-best_pos]

		if i == 0 and render_option is True: 
			save_name += '_' + str(chosen_index) 
			save_dir  += '_' + str(chosen_index)
			savepath = os.path.join('images', save_dir, save_name)
			renderer.composite(savepath, dataset.normalizer.unnormalize(np.expand_dims(np_samples[i], axis = 0), 'observations'))        

			traj_1_obs_visualize = dataset.normalizer.unnormalize(traj_1_obs[0:1, :, :], 'observations') 
			traj_2_obs_visualize = dataset.normalizer.unnormalize(traj_2_obs, 'observations') 
			savepath_traj_1 = os.path.join('images', save_dir, 'traj_1_full.png')      
			savepath_traj_2 = os.path.join('images', save_dir, 'traj_2_full.png') 

			renderer.composite(savepath_traj_1, traj_1_obs_visualize)  
			renderer.composite(savepath_traj_2, traj_2_obs_visualize)  

			traj_1_obs_visualize_partial = dataset.normalizer.unnormalize(traj_1_obs[0:1, :10,  :], 'observations') 
			traj_2_obs_visualize_partial = dataset.normalizer.unnormalize(traj_2_obs[0:1, :10,  :], 'observations') 
			savepath_traj_1 = os.path.join('images', save_dir, 'traj_1_partial.png')   
			savepath_traj_2 = os.path.join('images', save_dir, 'traj_2_partial.png')  
			renderer.composite(savepath_traj_1, traj_1_obs_visualize_partial)    
			renderer.composite(savepath_traj_2, traj_2_obs_visualize_partial)

	conditions_p = torch.tensor(cond_p).to(device)
	returns_p = to_device(test_ret * torch.ones(stitch_batch_size, 1), device)
	samples_p = trainer.ema_model.conditional_sample(conditions_p, returns=returns_p)  # shape is [64, 100, 11]
	np_samples_p = to_np(samples_p)     
	obss_p = dataset.normalizer.unnormalize(np_samples_p, 'observations')

	succ_cnt = 0 
	aug_list = [] 

	generated_transitions = 0 

	print(sum(mark_succ))

	total_number_traj = sum(mark_succ) 
	filtered_number_traj = 0

	for j in range(0, stitch_batch_size): 
		
		# failed to reach threshold 
		if mark_succ[j] < 0.5: 
			continue 
		end_index = positions[j][1]
		start_index = positions[j][0]
		best_pos = end_index + 1 
		len_path_data = end_index - start_index + 2
		return_info = {} 
		return_info['act'] = np.zeros(shape = (len_path_data, action_dim))
		return_info['rew'] = np.zeros(shape = (len_path_data))  
		return_info['obs'] = np.zeros(shape = (len_path_data + 1, obs_dim))

		normed_1 = [] 
		normed_2 = [] 
		normed_actions = [] 
		FLAG = False 
		for i in range(start_index, end_index + 2): 
			obs_comb = torch.cat([samples_p[j : j + 1, i - 1, :], samples_p[j : j + 1, i , :]], dim=-1)
			obs_comb = obs_comb.reshape(-1, 2 * obs_dim)          
			action = trainer.ema_model.inv_model(obs_comb) # [1,3]
			action = to_np(action)
			normed_actions.append(action) 
			continue 
		
		normed_actions = np.concatenate(normed_actions, axis = 0) 
		un_normed_actions = dataset.normalizer.unnormalize(normed_actions, 'actions') 

		for i in range(start_index, end_index + 2): 
			dynamics_info = dynamics_model.predict(
				torch.as_tensor(obss_p[j][i - 1], dtype = torch.float32).to(device), 
				torch.as_tensor(un_normed_actions[i - start_index], dtype = torch.float32).to(device).squeeze(), 
			)
			normed_1.append(np.expand_dims(obss_p[j][i], axis = 0))
			normed_2.append(np.expand_dims(dynamics_info['next_obs'], axis = 0))
			if dynamics_info['done'][0][0] == True: 
				print('Ends in {}, invalid!'.format(i))
				FLAG = True 
				break 
			return_info['rew'][i - start_index] =  dynamics_info['reward'][0][0]
			i_action = to_np(un_normed_actions[i - start_index])    
			return_info['act'][i - start_index] = i_action 

		if FLAG == True: 
			continue 
			
		normed_1 = np.concatenate(normed_1, axis = 0) 
		normed_2 = np.concatenate(normed_2, axis = 0) 

		normed_1 = dataset.normalizer.normalize(normed_1, 'observations') 
		normed_2 = dataset.normalizer.normalize(normed_2, 'observations') 

		for i in range(normed_1.shape[0]):
			if np.sum((normed_1[i] - normed_2[i]) ** 2) > dynamics_deviate: 
				FLAG = True 
				break 

		if FLAG == True: 
			filtered_number_traj += 1 
			continue 

		new_discounted_reward = expert_buffer.info[positions[j][2]][0]

		for k in range(return_info['rew'].shape[0] - 1, -1, -1): 
			new_discounted_reward = new_discounted_reward * 0.99 + return_info['rew'][k] 
		
		original_discounted_reward = expert_infos[j]['original_discounted_reward']
		print(f'{original_discounted_reward} -> {new_discounted_reward}')
		if original_discounted_reward > new_discounted_reward: 
			print('BAD CONCATE, WARNING\n')   
			continue 

		succ_cnt += 1
		return_info['obs'] = obss_p[j][start_index - 1: end_index + 2]   

		aug_cnt = return_info['rew'].shape[0]

		return_info['next_obs'] = return_info['obs'][1:]
		return_info['obs']      = return_info['obs'][:-1] 
		return_info['dones'] = np.full((aug_cnt,), False, dtype=bool) 

		for _ in ['obs', 'rew', 'dones', 'next_obs', 'act']:
			return_info[_] = return_info[_][:] 

		generated_transitions += return_info['obs'].shape[0]
		aug_list.append((return_info['obs'], return_info['act'], return_info['rew'], return_info['next_obs'], return_info['dones']))  

		if j == 0 and render_option == True: 
			savepath_p = os.path.join('images', save_dir, save_name + '_wd_total.png')
			renderer.composite(savepath_p, obss_p[0:1, :, :])
			savepath_comp = os.path.join('images', save_dir, save_name + '_wd_comp.png') 
			renderer.composite(savepath_comp, obss_p[0:1, : best_pos + 10 - 1,:])

	return aug_list, generated_transitions, (total_number_traj, filtered_number_traj)

class OptimalBuffer: 
	def __init__(self, ratio = 0.1, gamma = 0.99): 
		self.gamma = gamma
		self.ratio = ratio 
		self.info  = [] 
	def insert_traj(self, info): 
		current_total_reward = 0
		current_discounted_reward = 0
		for i in range(info['horizon'] - 1, -1, -1):
			current_total_reward += info['rew'][i] 
			current_discounted_reward = current_discounted_reward * self.gamma + info['rew'][i] 
			if info['horizon'] - i > 100: 
				current_info = {} 
				current_info['obs'] = info['obs'][i : i + 100] 
				self.info.append((current_discounted_reward, info['obs'][i : i + 100])) 
				continue  
	def sample_batch_traj(self, optim_batch, dataset): 
		# stored in (d-rtg, obs:(100, obs_dim)) 
		batch_index = random.sample(range(len(self.info)), optim_batch)     
		batch_info  = [self.info[_] for _ in batch_index]  
		first_obs = [_[1][0] for _ in batch_info] 
		first_obs = np.array(first_obs) 
		normalized_first_obs = dataset.normalizer.normalize(np.expand_dims(first_obs, axis = 0), 'observations')
		# (1, 512, 11) 

		for i in range(len(batch_info)): 
			info_i = batch_info[i] 
			info_update_i = [
				info_i[0], 
				info_i[1], 
				normalized_first_obs[0][i] 
			]
			batch_info[i] = info_update_i
		return batch_info, np.squeeze(normalized_first_obs, axis = 0)


class LoadSequenceDataset: 
	def __init__(self, env_name): 
		self.env_name = env_name 
		self.env = gym.make(env_name)
		dataset = self.env.get_dataset() 

		print(f"\nnumber of offline data is {dataset['observations'].shape[0]}\n")
		
		paths = []
		current_path = {'obs': [], 'act': [], 'rew': [], 'dones': [], 'next_obs': []}
		max_path_length = self.env._max_episode_steps   
		use_timeouts = 'timeouts' in dataset 

		print(use_timeouts) 

		for i in range(dataset['observations'].shape[0]):
			current_path['obs'].append(dataset['observations'][i])
			current_path['act'].append(dataset['actions'][i])
			current_path['rew'].append(dataset['rewards'][i])
			current_path['dones'].append(dataset['terminals'][i])
			current_path['next_obs'].append(dataset['next_observations'][i])   

			if use_timeouts:
				final_timestep = dataset['timeouts'][i]
			else:
				final_timestep = len(current_path['obs']) == max_path_length     

			if bool(dataset['terminals'][i]) or final_timestep: 
				for _ in current_path: 
					current_path[_] = np.array(current_path[_]) 
				paths.append(current_path)
				current_path = {'obs': [], 'act': [], 'rew': [], 'dones': [], 'next_obs': []}
		self.paths = paths 
		self.num_traj = len(self.paths) 

	def get_full_info_traj(self, idx, gamma = 0.99): 
		obs = [] 
		act = [] 
		rew = [] 
		next_obs = [] 
		dones = [] 
		obs = np.copy(self.paths[idx]['obs'])
		act = np.copy(self.paths[idx]['act'])
		rew = np.copy(self.paths[idx]['rew'])
		next_obs = np.copy(self.paths[idx]['next_obs'])
		dones = np.copy(self.paths[idx]['dones']) 
		total_return = np.sum(rew) 
		discounted_return = 0
		for i in range(rew.shape[0] - 1, -1, -1): 
			discounted_return = discounted_return * gamma + rew[i]
		return {
			'obs': obs, 'act': act, 'rew': rew, 'next_obs': next_obs, 
			'dones': dones, 'total_return': total_return, 'discounted_return': discounted_return, 
			'horizon': rew.shape[0]
		}
