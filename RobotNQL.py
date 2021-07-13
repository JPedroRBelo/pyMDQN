import torch
import torch.optim as optim
import numpy as np
import config as dcfg


class RobotNQL:
	def __init__(self,epi,cfg=dcfg,validation=False):
		#cpu or cuda
		self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.state_dim  = cfg.proc_frame_size #State dimensionality 84x84.
		self.actions	= cfg.actions
		self.n_actions  = len(self.actions)
		self.win=None
		#epsilon annealing
		self.ep_start   = cfg.ep_start
		self.ep		 = self.ep_start #Exploration probability.
		self.ep_end	 = cfg.ep_end
		self.ep_endt	= cfg.ep_endt
		self.learn_start= cfg.learn_start
		self.episode=epi

		if(validation):
			file_modelGray='validation/'+epi+'/modelGray.net'
			file_modelDepth='validation/'+epi+'/modelDepth.net'	
		else:
			file_modelGray='results/ep'+str(self.episode-1)+'/modelGray.net'
			file_modelDepth='results/ep'+str(self.episode-1)+'/modelDepth.net'	

		self.modelGray=torch.load(file_modelGray).to(self.device)
		self.modelDepth=torch.load(file_modelDepth).to(self.device)


	def perceive(self, state, depth, terminal, testing, numSteps, steps, testing_ep):
		curState = state.to(self.device)
		curDepth = depth.to(self.device)
		actionIndex = 0
		if not terminal:
			actionIndex = self.eGreedy(curState,curDepth, numSteps, steps, testing_ep)
			print("Action: ",self.actions[actionIndex])
			return actionIndex
		else:
			return 0
	'''
	def select_action(state):
		global steps_done
		sample = random.random()
		eps_threshold = EPS_END + (EPS_START - EPS_END) * \
			math.exp(-1. * steps_done / EPS_DECAY)
		steps_done += 1
		if sample > eps_threshold:
			with torch.no_grad():
			# t.max(1) will return largest column value of each row.
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
				return self.modelGray(state).max(1)[1].view(1, 1)
		else:
			return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
	'''

	def eGreedy(self,state,depth, numSteps , steps, testing_ep):
		self.ep = testing_ep or (self.ep_end +
			max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
			max(0, numSteps - self.learn_start))/self.ep_endt))
		print('Exploration probability ',self.ep)
		#-- Epsilon greedy

		if torch.rand(1) < self.ep:
			return np.random.randint(0, self.n_actions)
		else:
			return self.greedy(state,depth)
		
	def greedy(self,state,depth):
		print("greedy")
		self.modelGray.eval()
		self.modelDepth.eval()
		#state = state:float()
		#depth=depth:float()
		win=None
		q1 = self.modelGray.forward(state).cpu().detach().numpy()[0]
		q2 = self.modelDepth.forward(depth).cpu().detach().numpy()[0]
		ts = 0
		td = 0
		for i in range(self.n_actions):
			ts += q1[i]
			td += q2[i]
		q_fus=(q1/ts)*0.5+(q2/td)*0.5
		maxq = q_fus[0]
		besta = [0]
		for a in range(1, self.n_actions):
			if q_fus[a] > maxq:
				besta = [a]
				maxq = q_fus[a]
			elif q_fus[a] == maxq:
				besta.append(a)
		self.bestq = maxq
		r = np.random.randint(0,len(besta))
		self.lastAction = besta[r]
		return besta[r]
