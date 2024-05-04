import argparse
import gymnasium as gym
import numpy as np
import os
import torch

import BCQ
import utils
from cus_env import CustomEnv
import matplotlib
import matplotlib.pyplot as plt

# Trains BCQ offline
def train_BCQ(obs_dim, action_dim, max_action, device, args):
	# For saving files
	setting = f"{args.env}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize policy
	policy = BCQ.BCQ(obs_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

	# Load buffer
	replay_buffer = utils.ReplayBuffer(obs_dim, action_dim, device)
	replay_buffer.load(f"./(0.01)_buffer/{buffer_name}")
	
	evaluations = []
	episode_num = 0
	done = True 
	training_iters = 0
	value = 0
	while training_iters < args.max_timesteps: 
		pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
		value = eval_policy(policy,training_iters,10)

		evaluations.append(value)
		np.save(f"./results/modelled_data_{setting}", evaluations)
		
		plt.plot(value,training_iters)
		training_iters += args.eval_freq
		print(training_iters)
		print(f"Training iterations: {training_iters}")
	plt.show()
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy,num,eval_episodes=10):
	eval_env = CustomEnv()
	avg_reward = 0.
	directory = "./modelled_fig"
	for val in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
	plt.savefig(directory + f'/{num}_bcq_.png')

	avg_reward /= eval_episodes
	
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="cus_env")               # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
	parser.add_argument("--eval_freq", default=500, type=float)     # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e4, type=int)   # Max time steps to run environment or train for (this defines buffer size)
	parser.add_argument("--start_timesteps", default=2500, type=int)# Time steps initial random policy is used before training behavioral
	parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
	parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
	parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
	parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
	#parser.add_argument("--train_behavioral")  # If true, train behavioral (DDPG)
	#parser.add_argument("--generate_buffer")   # If true, generate buffer
	args = parser.parse_args()

	print("---------------------------------------")	
	
	print(f"Setting: Training BCQ, Env: {args.env}")
	print("---------------------------------------")


	env = CustomEnv()
	
	obs_dim = env.obs_dim
	state_dim = env.state_dim
	action_dim = env.act_dim 
	y_ref_dim = env.y_ref_dim
	max_action = float(env.action_space.high[0])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_BCQ(obs_dim, action_dim, max_action, device, args)