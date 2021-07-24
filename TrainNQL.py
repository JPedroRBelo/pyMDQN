import torch
import torch.optim as optim
import numpy as np
import os
import re
import copy
import random
from network import DQN
from collections import namedtuple
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sys import getsizeof
import config as dcfg #defaultConfig

Transition = namedtuple('Transition',
                        ('sgray','sdepth','action','next_sgray','next_sdepth','reward'))


class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.aux_memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
		    self.memory.append(None)
		    self.aux_memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.aux_memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		replay_sample = random.sample(self.memory, batch_size)
		return replay_sample

	def pull(self,batch_size):
		if(batch_size>len(self.aux_memory)):
			self.aux_memory = self.memory.copy();
		replay_sample = [self.aux_memory.pop(random.randrange(len(self.aux_memory))) for _ in range(batch_size)]
		return replay_sample

	def __len__(self):
		return len(self.memory)


class TrainNQL:
	def __init__(self,epi,cfg=dcfg,validation=False):
		#cpu or cuda
		torch.cuda.empty_cache()
		self.device = cfg.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.state_dim  = cfg.proc_frame_size #State dimensionality 84x84.
		self.state_size = cfg.state_size
		#self.t_steps= tsteps
		self.t_eps = cfg.t_eps
		self.minibatch_size = cfg.minibatch_size
		# Q-learning parameters
		self.discount       = cfg.discount #Discount factor.
		self.replay_memory  = cfg.replay_memory
		self.bufferSize     =  cfg.bufferSize
		self.target_q       = cfg.target_q
		self.validation = validation
		if(validation):
			self.episode = epi
		else:
			self.episode=int(epi)-1
		self.cfg = cfg

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
			self.gray_policy_net =  DQN(noutputs=cfg.noutputs,nfeats=cfg.nfeats,nstates=cfg.nstates,kernels=cfg.kernels,strides=cfg.strides,poolsize=cfg.poolsize).to(self.device)
			self.gray_target_net =  DQN(noutputs=cfg.noutputs,nfeats=cfg.nfeats,nstates=cfg.nstates,kernels=cfg.kernels,strides=cfg.strides,poolsize=cfg.poolsize).to(self.device)
			self.depth_policy_net = DQN(noutputs=cfg.noutputs,nfeats=cfg.nfeats,nstates=cfg.nstates,kernels=cfg.kernels,strides=cfg.strides,poolsize=cfg.poolsize).to(self.device)
			self.depth_target_net =  DQN(noutputs=cfg.noutputs,nfeats=cfg.nfeats,nstates=cfg.nstates,kernels=cfg.kernels,strides=cfg.strides,poolsize=cfg.poolsize).to(self.device)
			

		if not validation and self.target_q and self.episode % self.target_q == 0:
			print ("cloning")
			self.gray_target_net=copy.deepcopy(self.gray_policy_net)
			self.depth_target_net=copy.deepcopy(self.depth_policy_net)

		self.gray_target_net.load_state_dict(self.gray_target_net.state_dict())
		self.gray_target_net.eval()

		
		self.depth_target_net.load_state_dict(self.depth_target_net.state_dict())
		self.depth_target_net.eval()


		self.gray_optimizer = optim.RMSprop(self.gray_policy_net.parameters())
		self.depth_optimizer = optim.RMSprop(self.depth_policy_net.parameters())
		self.memory = ReplayMemory(self.replay_memory)

	def get_tensor_from_image(self,file):
		convert = T.Compose([T.ToPILImage(),
			T.Resize((self.state_dim,self.state_dim), interpolation=Image.BILINEAR),
			T.ToTensor()])
		screen = Image.open(file)
		screen = np.ascontiguousarray(screen, dtype=np.float32)/255
		screen = torch.from_numpy(screen)
		screen = convert(screen).unsqueeze(0).to(self.device)
		return screen

	def get_data(self,episode,tsteps):	
		#images=torch.Tensor(tsteps,self.state_size,self.state_dim,self.state_dim).to(self.device)
		#depths=torch.Tensor(tsteps,self.state_size,self.state_dim,self.state_dim).to(self.device)
		images = []
		depths = []
		dirname_rgb='dataset/RGB/ep'+str(episode)
		dirname_dep='dataset/Depth/ep'+str(episode)
		for step in range(tsteps):
			#proc_image=torch.Tensor(self.state_size,self.state_dim,self.state_dim).to(self.device)
			#proc_depth=torch.Tensor(self.state_size,self.state_dim,self.state_dim).to(self.device)
			proc_image = []
			proc_depth = []

			dirname_rgb='dataset/RGB/ep'+str(episode)
			dirname_dep='dataset/Depth/ep'+str(episode)
			for i in range(self.state_size):
				grayfile=dirname_rgb+'/image_'+str(step+1)+'_'+str(i+1)+'.png'
				depthfile=dirname_dep+'/depth_'+str(step+1)+'_'+str(i+1)+'.png'
				#proc_image[i] = self.get_tensor_from_image(grayfile)
				#proc_depth[i] = self.get_tensor_from_image(depthfile)	
				proc_image.append(grayfile)
				proc_depth.append(depthfile)	
			#images[step]=proc_image
			#depths[step]=proc_depth
			images.append(proc_image)
			depths.append(proc_depth)	
		return images,depths	

	def load_data(self):

		rewards=torch.load('files/reward_history.dat')
		actions=torch.load('files/action_history.dat')
		ep_rewards=torch.load('files/ep_rewards.dat')

		print("Loading images")

		best_scores = range(len(actions))
		
		eps_values = []
		for i in range(len(actions)):

			hspos = 0
			hsneg = 0			
			for step in range(len(actions[i])):		
				if(len(actions[i])>0 ):
					if actions[i][step] == 3 :
						if rewards[i][step]==self.cfg.hs_success_reward:
							hspos = hspos+1
						elif rewards[i][step]==self.cfg.hs_fail_reward : 
							hsneg = hsneg+1	
			accuracy = float(((hspos)/(hspos+hsneg)))			
			eps_values.append(accuracy)

		best_scores = np.argsort(eps_values)


			


		for i in best_scores:
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

			#os.system("free -h")
			#with torch.no_grad():
			images,depths=self.get_data(i+1,k)	
			print ("Loading done")
			
			
			for step  in range(k-1):

				reward = rewards[i][step]

				'''
				reward = self.cfg.neutral_reward
				if rewards[i][step]>=1:
					reward = self.cfg.hs_success_reward
				elif rewards[i][step]<0:
					reward = self.cfg.hs_fail_reward				'''

				reward = torch.tensor([reward], device=self.device)
				action = torch.tensor([[actions[i][step]]], device=self.device, dtype=torch.long)
				#image = images[step].unsqueeze(0).to(self.device)
				#depth = depths[step].unsqueeze(0).to(self.device)
				#next_image = images[step+1].unsqueeze(0).to(self.device)
				#next_depth = depths[step+1].unsqueeze(0).to(self.device)
				image = images[step]
				depth = depths[step]
				next_image = images[step+1]
				next_depth = depths[step+1]
				self.memory.push(image,depth,action,next_image,next_depth,reward)
				#print("Memory size: ",getsizeof(self.memory))
				#torch.cuda.empty_cache()	


	def train(self):
		if self.bufferSize < self.minibatch_size:
		    return
		for i in range(0,self.bufferSize,self.minibatch_size):
			#transitions = self.memory.sample(self.minibatch_size)
			transitions = self.memory.pull(self.minibatch_size)

			print('Batch train: '+str(int(i/self.minibatch_size)+1)+"/"+str(int(self.bufferSize/self.minibatch_size)+1))
			
			aux_transitions = []
			for t in transitions:
				proc_sgray=torch.Tensor(self.state_size,self.state_dim,self.state_dim).to(self.device)
				proc_sdepth=torch.Tensor(self.state_size,self.state_dim,self.state_dim).to(self.device)
				proc_next_sgray=torch.Tensor(self.state_size,self.state_dim,self.state_dim).to(self.device)
				proc_next_sdepth=torch.Tensor(self.state_size,self.state_dim,self.state_dim).to(self.device)
				count = 0
				for sgray,sdepth,next_sgray,next_sdepth in zip(t.sgray,t.sdepth,t.next_sgray,t.next_sdepth):
					proc_sgray[count] = self.get_tensor_from_image(sgray)
					proc_sdepth[count] = self.get_tensor_from_image(sdepth)	
					proc_next_sgray[count] = self.get_tensor_from_image(next_sgray)
					proc_next_sdepth[count] = self.get_tensor_from_image(next_sdepth)	
					count += 1
				
				proc_sgray = proc_sgray.unsqueeze(0).to(self.device)
				proc_sdepth = proc_sdepth.unsqueeze(0).to(self.device)
				proc_next_sgray = proc_next_sgray.unsqueeze(0).to(self.device)
				proc_next_sdepth = proc_next_sdepth.unsqueeze(0).to(self.device)
				#('sgray','sdepth','action','next_sgray','next_sdepth','reward')
				one_transition = Transition(proc_sgray,proc_sdepth,t.action,proc_next_sgray,proc_next_sdepth,t.reward)
				aux_transitions.append(one_transition)
			transitions = aux_transitions

			# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
			# detailed explanation). This converts batch-array of Transitions
			# to Transition of batch-arrays.
			batch = Transition(*zip(*transitions))
			#print(batch.sgray)

			# Compute a mask of non-final states and concatenate the batch elements
			# (a final state would've been the one after which simulation ended)
			gray_non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
			                                      batch.next_sgray)), device=self.device, dtype=torch.bool)
			gray_non_final_next_states = torch.cat([s for s in batch.next_sgray
			                                            if s is not None])

			depth_non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
			                                      batch.next_sdepth)), device=self.device, dtype=torch.bool)
			depth_non_final_next_states = torch.cat([s for s in batch.next_sdepth
			                                            if s is not None])
			sgray_batch = torch.cat(batch.sgray)
			sdepth_batch = torch.cat(batch.sdepth)

			action_batch = torch.cat(batch.action)
			reward_batch = torch.cat(batch.reward)

			# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
			# columns of actions taken. These are the actions which would've been taken
			# for each batch state according to policy_net
			sgray_action_values = self.gray_policy_net(sgray_batch).gather(1, action_batch)
			sdepth_action_values = self.depth_policy_net(sdepth_batch).gather(1, action_batch)

			# Compute V(s_{t+1}) for all next states.
			# Expected values of actions for non_final_next_states are computed based
			# on the "older" target_net; selecting their best reward with max(1)[0].
			# This is merged based on the mask, such that we'll have either the expected
			# state value or 0 in case the state was final.
			next_sgray_values = torch.zeros(self.minibatch_size, device=self.device)
			next_sgray_values[gray_non_final_mask] = self.gray_target_net(gray_non_final_next_states).max(1)[0].detach()


			next_sdepth_values = torch.zeros(self.minibatch_size, device=self.device)
			next_sdepth_values[depth_non_final_mask] = self.depth_target_net(depth_non_final_next_states).max(1)[0].detach()
			# Compute the expected Q values
			expected_sgray_action_values = (next_sgray_values * self.discount) + reward_batch
			expected_sdepth_action_values = (next_sdepth_values * self.discount) + reward_batch

			# Compute Huber loss
			gray_loss = F.smooth_l1_loss(sgray_action_values, expected_sgray_action_values.unsqueeze(1))
			depth_loss = F.smooth_l1_loss(sdepth_action_values, expected_sdepth_action_values.unsqueeze(1))

			# Optimize the model
			self.gray_optimizer.zero_grad()
			gray_loss.backward()
			for param in self.gray_policy_net.parameters():
			    param.grad.data.clamp_(-1, 1)
			self.gray_optimizer.step()

			# Optimize the model
			self.depth_optimizer.zero_grad()
			depth_loss.backward()
			for param in self.depth_policy_net.parameters():
			    param.grad.data.clamp_(-1, 1)
			self.depth_optimizer.step()


