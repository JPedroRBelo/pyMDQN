import torch
import torch.nn as nn
import torch.nn.functional as F




gpu=1
noutputs=4
nfeats=8
nstates=[16,32,64,256]
#kernel={9,5}
kernel1 = 4
kernel2 = 2
stride1 = 3
stride2 = 1
poolsize=2

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels=nfeats,out_channels=nstates[0], kernel_size=kernel1,stride=stride1,padding=1),
			nn.ReLU(),
			nn.MaxPool2d(poolsize),
			nn.Conv2d(in_channels=nstates[0],out_channels=nstates[1], kernel_size=kernel2,stride=stride2),
			nn.ReLU(),
			nn.MaxPool2d(poolsize),
			nn.Conv2d(in_channels=nstates[1],out_channels=nstates[2], kernel_size=kernel2,stride=stride2),
			nn.ReLU(),
			nn.MaxPool2d(poolsize),	
			)
		self.classifier = nn.Sequential(
			nn.Linear(nstates[2]*kernel2*kernel2,nstates[3]),
			nn.ReLU(),
			nn.Linear(nstates[3], noutputs),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(nstates[2]*kernel2*kernel2)
		x = self.classifier(x)
		return x