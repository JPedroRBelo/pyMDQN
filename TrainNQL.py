import torch
import torch.optim as optim
import numpy as np
import os
import re
from gmodel import DQN
from train_environment import get_data
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('sgray','sdepth' 'action','reward'))


class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
		    self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class TrainNQL:
	def __init__(self,epi,tsteps):
		#cpu or cuda
		self.device = "cuda" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.state_dim  = 84 #State dimensionality 84x84.
		self.actions	= ['1','2','3','4']
		self.n_actions  = len(self.actions)
		

		self.t_steps= tsteps
		self.t_eps = 30

		#epsilon annealing
		self.ep_start   = 1
		self.ep         = self.ep_start #Exploration probability.
		self.ep_end     = 0.1
		#self.ep_endt    = self.t_eps*self.t_steps#30000
		self.ep_endt    = 2000

		#learning rate annealing
		self.lr_start       = 0.00025 #Learning rate.
		self.lr             = self.lr_start
		self.lr_end         = self.lr
		#self.lr_endt        = 100000   #replay memory size
		self.wc             = 0  # L2 weight cost.
		self.minibatch_size = 25
		self.valid_size     = 500

		# Q-learning parameters
		self.discount       = 0.99 #Discount factor.


		self.numSteps = 0
		# Number of points to replay per learning step.
		self.n_replay       = 1
		# Number of steps after which learning starts.
		self.learn_start    = 0
		# Size of the transition table.
		#self.replay_memory  = 14000--10000
		#self.replay_memory  = self.t_eps*self.t_steps
		self.replay_memory  = 60000


		self.hist_len       = 8
		self.clip_delta     = 1
		self.target_q       = 4
		self.bestq          = 0

		self.gpu            = 1

		self.ncols          = 1  #number of color channels in input
		self.input_dims     = [8, self.state_dim, self.state_dim]
		self.histType       = "linear"  # history type to use
		self.histSpacing    = 1

		self.bufferSize     =  2000

		self.episode=epi-1

		modelGray='results/ep'+str(self.episode)+'/modelGray.net'
		modelDepth='results/ep'+str(self.episode)+'/modelDepth.net'
		tModelGray='results/ep'+str(self.episode)+'/tModelGray.net'
		tModelDepth='results/ep'+str(self.episode)+'/tModelDepth.net'


		if os.path.exists(modelGray) and os.path.exists(modelDepth):
			print("Loading model")
			self.gray_policy_net=torch.load(modelGray).to(self.device)
			self.gray_target_net=torch.load(tModelGray).to(self.device)
			self.depth_policy_net=torch.load(modelDepth).to(self.device)
			self.depth_target_net=torch.load(tModelDepth).to(self.device)

		else:
			print("New model")
			self.gray_policy_net = DQN().to(self.device)
			self.gray_target_net = DQN().to(self.device)
			self.depth_policy_net = DQN().to(self.device)
			self.depth_target_net = DQN().to(self.device)
			

		if self.target_q and self.episode % self.target_q == 0:
			print ("cloning")
			self.depth_policy_net = DQN().to(self.device)
			self.depth_target_net = DQN().to(self.device)
		
		self.gray_target_net.load_state_dict(self.gray_target_net.state_dict())
		self.gray_target_net.eval()

		
		self.depth_target_net.load_state_dict(self.depth_target_net.state_dict())
		self.depth_target_net.eval()


		self.gray_optimizer = optim.RMSprop(self.gray_policy_net.parameters())
		self.depth_optimizer = optim.RMSprop(self.depth_policy_net.parameters())
		self.memory = ReplayMemory(self.replay_memory)


	def load_data(self):
		print("Loading images")

		for i in range(self.t_eps):
			print('Ep: ',i+1)
			dirname_gray='dataset/RGB/ep'+str(i+1)
			dirname_dep='dataset/Depth/ep'+str(i+1)
			files = []
			if(os.path.exists(dirname_gray)):
				files = os.listdir(dirname_gray)

			k = 0
			for file in files:
				if re.match(r"image.*\.png", file):
					k=k+1
			k = int(k/8)
			while(k%4!=0):
				k = k-1
			if(k>self.bufferSize):
				k = self.bufferSize
			print(k)


		
			images=torch.Tensor(k,self.hist_len,self.state_dim,self.state_dim)
			depths=torch.Tensor(k,self.hist_len,self.state_dim,self.state_dim)	
	
			images,depths=get_data(i+1,k)	
			print ("Loading done")
			aset = ['1','2','3','4']
	
			rewards=torch.load('files/reward_history.dat')
			actions=torch.load('files/action_history.dat')
			ep_rewards=torch.load('files/ep_rewards.dat')
			for step  in range(k):
				terminal = 0
				if rewards[i][step]>3:
					self.memory.push(images[step],depths[step],actions[i][step],1)
				elif rewards[i][step]<0:
					self.memory.push(images[step],depths[step],actions[i][step],-0.1)
				else:
					self.memory.push(images[step],depths[step],actions[i][step],0)


	def train(self):
		pass
		#self.network_A.train()
		#self.network_B.train()





train = TrainNQL(1,10)
train.load_data()