import torch
import torch.optim as optim
import numpy as np


class RobotNQL:
	def __init__(self,epi):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.state_dim  = 84 #State dimensionality 84x84.
		self.actions	= {'1','2','3','4'}
		self.n_actions  = len(self.actions)
		self.win=None
		#epsilon annealing
		self.ep_start   = 1.0
		self.ep		 = self.ep_start #Exploration probability.
		self.ep_end	 = 0.1
		self.ep_endt	= 60000
		self.learn_start= 0


		#self.bufferSize =  2000
		self.episode=epi-1
		self.iter=1
		self.seq=""	

		modelA='results/ep'+str(self.episode)+'/modelA_cuda.net'
		modelB='results/ep'+str(self.episode)+'/modelB_cuda.net'

		print(modelA)
		print(modelB)

		self.networkA=torch.load(modelA).to(self.device)
		self.networkB=torch.load(modelB).to(self.device)

		self.numSteps = 0 #Number of perceived states.
		self.lastState = None
		self.lastDepth = None
		self.lastAction = None
		self.lastTerminal = None


	def perceive(self,reward, state, depth, terminal, testing, numSteps, steps, testing_ep):
		curState = state
		curDepth = depth  
		actionIndex = 0
		if not terminal:
			actionIndex = self.eGreedy(curState,curDepth, numSteps, steps, testing_ep)
			return actionIndex
		else:
			return 0

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
		self.networkA.eval()
		self.networkB.eval()
		#state = state:float()
		#depth=depth:float()
		win=None
		#q1 = np.array([0.20,0.40,0.20,0.40])
		#q2 = np.array([0.20,0.40,0.20,0.40])
		q1 = self.networkA.forward(state)
		q2 = self.networkB.forward(depth)
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
