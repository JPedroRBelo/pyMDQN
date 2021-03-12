import torch
import torch.optim as optim
import numpy as np
import os
from gmodel import Net

class TrainNQL:
	def __init__(self,epi,tsteps):
		#cpu or cuda
		self.device = "gpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.state_dim  = 84 #State dimensionality 84x84.
		self.actions	= {'1','2','3','4'}
		self.n_actions  = len(self.actions)
		

		self.t_steps= tsteps
		self.t_eps = 30

		#epsilon annealing
		self.ep_start   = 1
		self.ep         = self.ep_start #Exploration probability.
		self.ep_end     = 0.1
		#self.ep_endt    = self.t_eps*self.t_steps#30000
		self.ep_endt    = 60000

		#learning rate annealing
		self.lr_start       = 0.00025 #Learning rate.
		self.lr             = self.lr_start
		self.lr_end         = self.lr
		self.lr_endt        = 100000   #replay memory size
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

		modelA='results/ep'+str(self.episode)+'/modelA_cpu.net'
		modelB='results/ep'+str(self.episode)+'/modelB_cpu.net'
		tmodelA='results/ep'+str(self.episode)+'/tmodelA_cpu.net'
		tmodelB='results/ep'+str(self.episode)+'/tmodelB_cpu.net'

		if(self.device=="cuda"):
			modelA='results/ep'+str(self.episode)+'/modelA_cuda.net'
			modelB='results/ep'+str(self.episode)+'/modelB_cuda.net'
			tmodelA='results/ep'+str(self.episode)+'/tmodelA_cuda.net'
			tmodelB='results/ep'+str(self.episode)+'/tmodelB_cuda.net'

		if os.path.exists(modelA) and os.path.exists(modelB):
			print("Loading model")
			# y-channel	
			self.network_A=torch.load(modelA)
			self.target_network_A=torch.load(tmodelA)
			#depth channel
			self.network_B=torch.load(modelB)
			self.target_network_B=torch.load(tmodelB)
		else:
			print("New model")
			self.network_A = Net()
			self.network_B = Net()
			self.target_network_A = Net()
			self.target_network_B = Net()			

		if self.target_q and self.episode % self.target_q == 0:
			print ("cloning")
			self.target_network_A = self.network_A.clone()
			self.target_network_B = self.network_B.clone()


train = TrainNQL(1,10)