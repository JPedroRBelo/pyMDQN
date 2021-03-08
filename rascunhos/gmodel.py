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




def create_network():
	modelA = torch.nn.Sequential()
	#conv1
	modelA.add_module("conv1", nn.Conv2d(in_channels=nfeats,out_channels=nstates[0], kernel_size=kernel1,stride=stride1,padding=1))
	modelA.add_module("relu1", nn.ReLU())
	modelA.add_module("maxpool1", nn.MaxPool2d(poolsize))
	#conv2
	modelA.add_module("conv2", nn.Conv2d(in_channels=nstates[0],out_channels=nstates[1], kernel_size=kernel2,stride=stride2))
	modelA.add_module("relu2", nn.ReLU())
	modelA.add_module("maxpool2", nn.MaxPool2d(poolsize))
	#conv3
	modelA.add_module("conv3", nn.Conv2d(in_channels=nstates[1],out_channels=nstates[2], kernel_size=kernel2,stride=stride2))
	modelA.add_module("relu3", nn.ReLU())
	modelA.add_module("maxpool3", nn.MaxPool2d(poolsize))
	#RESHAPE
	modelA.add_module("view",nn.View(nstates[2]*kernel2*kernel2))
	return modelA#,modelB

	'''
	#RESHAPE
	modelA:add(nn.View(nstates[3]*kernel[2]*kernel[2]))
	modelA:add(nn.Linear(nstates[3]*kernel[2]*kernel[2],nstates[4]))
	modelA:add(nn.ReLU())
	modelA:add(nn.Linear(nstates[4],noutputs))
	modelA=modelA:cuda()


	modelB=nn.Sequential()
	#cov1
	modelB:add(nn.SpatialConvolution(nfeats, nstates[1],kernel[1],kernel[1],stride[1],stride[1],1))
	modelB:add(nn.ReLU())
	modelB:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	#cov2
	modelB:add(nn.SpatialConvolution(nstates[1],nstates[2],kernel[2],kernel[2],stride[2],stride[2]))
	modelB:add(nn.ReLU())
	modelB:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	#cov3
	modelB:add(nn.SpatialConvolution(nstates[2],nstates[3],kernel[2],kernel[2],stride[2],stride[2]))
	modelB:add(nn.ReLU())
	modelB:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	#RESHAPE
	modelB:add(nn.View(nstates[3]*kernel[2]*kernel[2]))
	modelB:add(nn.Linear(nstates[3]*kernel[2]*kernel[2],nstates[4]))
	modelB:add(nn.ReLU())

	modelB:add(nn.Linear(nstates[4],noutputs))

	modelB=modelB:cuda()

	'''

	


net = create_network()
print(net)