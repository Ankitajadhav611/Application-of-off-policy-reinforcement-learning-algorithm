import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim ,device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def save(self, save_folder):
		np.save(f"./modelled_buffer/state.npy", self.state[:self.size])
		np.save(f"./modelled_buffer/action.npy", self.action[:self.size])
		np.save(f"./modelled_buffer/next_state.npy", self.next_state[:self.size])
		np.save(f"./modelled_buffer/reward.npy", self.reward[:self.size])
		np.save(f"./modelled_buffer/not_done.npy", self.not_done[:self.size])
		np.save(f"./modelled_buffer/ptr.npy", self.ptr)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"./modelled_buffer/reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer.shape[0], size)

		self.state[:self.size] = np.load(f"./modelled_buffer/state.npy")[:self.size]
		self.action[:self.size] = np.load(f"./modelled_buffer/action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"./modelled_buffer/next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer[:self.size]
		self.not_done[:self.size] = np.load(f"./modelled_buffer/not_done.npy")[:self.size]