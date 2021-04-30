import torch
import torch.nn as nn
import torch.nn.functional as F

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
			nn.BatchNorm2d(nstates[0]),
			nn.ReLU(),
			nn.MaxPool2d(poolsize),
			nn.Conv2d(in_channels=nstates[0],out_channels=nstates[1], kernel_size=kernel2,stride=stride2),
			nn.BatchNorm2d(nstates[1]),
			nn.ReLU(),
			nn.MaxPool2d(poolsize),
			nn.Conv2d(in_channels=nstates[1],out_channels=nstates[2], kernel_size=kernel2,stride=stride2),
			nn.BatchNorm2d(nstates[2]),
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
		x = x.view(x.size(0),nstates[2]*kernel2*kernel2)
		x = self.classifier(x)
		return x


'''

h = 84
w = 84
class DQN(nn.Module):

	def __init__(self):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(nfeats, nstates[0], kernel_size=kernel1, stride=stride1)
		self.bn1 = nn.BatchNorm2d(nstates[0])
		self.conv2 = nn.Conv2d(nstates[0], nstates[1], kernel_size=kernel2, stride=stride2)
		self.bn2 = nn.BatchNorm2d(nstates[1])
		self.conv3 = nn.Conv2d(nstates[1], nstates[2], kernel_size=kernel2, stride=stride2)
		self.bn3 = nn.BatchNorm2d(nstates[2])

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size = kernel1, stride = stride1):
			return (size - (kernel_size - 1) - 1) // stride  + 1
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		print(convw,convh)
		linear_input_size = convw * convh * nstates[2]
		self.head = nn.Linear(linear_input_size, noutputs)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		print(x.shape)
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

'''


