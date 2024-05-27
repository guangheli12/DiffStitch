
import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np 
import gym 
import d4rl 
from models.transition_model import TransitionModel
# from scripts.mopo.models.transition_model import TransitionModel
import numpy as np
import pickle 
class StaticFnsHopper:
    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = np.isfinite(next_obs).all(axis=-1) \
                   * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                   * (height > .7) \
                   * (np.abs(angle) < .2)

        done = ~not_done
        return done

class StaticFnsWalker2d: 
    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = (height > 0.8) \
                   * (height < 2.0) \
                   * (angle > -1.0) \
                   * (angle < 1.0)
        done = ~not_done
        return done

class StaticFnsCheetah:
    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        done = np.array([False]).repeat(len(obs))
        return done


# hopper-medium-expert-v2
class RewardPredictingModel:
    # /root/autodl-tmp/liguanghe/workspace/DiffRL/diff-2/code/scripts/mopo
    def __init__(self, 
        device = torch.device('cuda:0'), 
        env_name = 'walker2d-medium-replay-v2', 
        load_path = ''
    ): 
        self.env_name = env_name
        env = gym.make(env_name)
        transition_params = {
            "model_batch_size": 256,
            "use_weight_decay": True,
            "optimizer_class": "Adam",
            "learning_rate": 0.001,
            "holdout_ratio": 0.2,
            "inc_var_loss": True,
            "model": {
                "hidden_dims": [200, 200, 200, 200],
                "decay_weights": [0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
                "act_fn": "swish",
                "out_act_fn": "identity",
                "num_elite": 5,
                "ensemble_size": 7
            }
        }
        self.device = device 
        # create dynamics model
        if 'walker2d' in env_name:
            self.dynamics_model = TransitionModel(
                                obs_space = env.observation_space,
                                action_space = env.action_space,
                                static_fns = StaticFnsWalker2d,
                                lr = 0.001,
                                **transition_params
                            )
        if 'hopper' in env_name: 
            self.dynamics_model = TransitionModel(
                                obs_space = env.observation_space,
                                action_space = env.action_space,
                                static_fns = StaticFnsHopper, 
                                lr = 0.001,
                                **transition_params
                            )
        if 'cheetah' in env_name: 
            self.dynamics_model = TransitionModel(
                                obs_space = env.observation_space,
                                action_space = env.action_space,
                                static_fns = StaticFnsCheetah, 
                                lr = 0.001,
                                **transition_params
                            )
        self.load_path = load_path 

        for network_name, network in self.dynamics_model.networks.items(): 
            save_path = os.path.join(load_path, network_name + ".pt")
            self.dynamics_model.model = torch.load(save_path, map_location = self.device)
            self.dynamics_model.networks["model"] = self.dynamics_model.model

    
    def predict(self, obs, act): 
        pred = self.dynamics_model.predict(obs, act, deterministic = True) 
        return { 'next_obs': pred[0], 'reward':  pred[1], 'done': pred[2]}
        



# debug 
# if __name__ == '__main__': 
#     reward_model = RewardPredictingModel()   
#     import gym
#     import d4rl
#     import pickle
#     path = '/root/aug_data/hop_med_exp/hop_med_exp_3.pkl'
#     with open(path, "rb") as f: 
#         data = pickle.load(f) 

#     env = gym.make('hopper-medium-expert-v2')
#     dataset = env.get_dataset() 
#     device = torch.device('cuda:0')

#     for i in range(2000): 
#         print('\n\n---------------------------\n\n')
#         print(dataset['rewards'][i])
#         rew  = reward_model.predict(
#             torch.tensor(dataset['observations'][i], dtype = torch.float32).to(device),
#             torch.tensor(dataset['actions'][i], dtype = torch.float32).to(device)
#         )
#         print(rew['reward']) 
#         """
#         rew = reward_model.predict(
#             torch.tensor(obs[i], dtype = torch.float32).to(device), 
#             torch.tensor(act[i], dtype = torch.float32).to(device)
#         ) 

#         print('\n\n-----------------\n\n')
#         # print(rew['next_obs'])
#         # print(obs[i + 1])
#         print(s_rew[i], rew['reward']) 
#         """
#     sys.exit(0)
#     print(dataset['observations'].shape)
#     s_rew = data[5][2]
#     obs = data[5][0] 
#     act = data[5][1]
#     print(obs.shape) 
#     print(act.shape) 
#     # print(re)

#     for i in range(obs.shape[0] - 1): 
#         rew = reward_model.predict(
#             torch.tensor(obs[i], dtype = torch.float32).to(device), 
#             torch.tensor(act[i], dtype = torch.float32).to(device)
#         ) 

#         print('\n\n-----------------\n\n')
#         # print(rew['next_obs'])
#         # print(obs[i + 1])
#         print(s_rew[i], rew['reward']) 
#     pass 

