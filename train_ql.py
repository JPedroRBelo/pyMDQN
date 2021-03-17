import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from train_environment import get_data
from pathlib import Path
import copy



t_steps=2050
device = "gpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(torch.initial_seed())  
win=None

episode=torch.load('files/episode.dat')
agent=TrainNQL(epi=episode, tsteps=t_steps)

win=None

def main():
	target_net=4
	agent.load_data()
	for j in range(50):
		print("\nTrain="..j.."/50")
		for i in range(10):
			print("epoch")
			agent.train()
		agent.target_network_A=copy.deepcopy(agent.network_A)
		agent.target_network_B=copy.deepcopy(agent.network_B)	


	modelA=copy.deepcopy(agent.network_A)
	modelB=copy.deepcopy(agent.network_B)

	if episode%target_net==1:
		agent.target_network_A=copy.deepcopy(agent.network_A)
		agent.target_network_B=copy.deepcopy(agent.network_B)

  	tmodelA = copy.deepcopy(agent.target_network_A)
  	tmodelB = copy.deepcopy(agent.target_network_B)
	model_dir='results/ep'+str(episode)
	Path(model_dir).mkdir(parents=True, exist_ok=True)

 	'''
	save_modelA_gpu=model_dir..'/modelA_gpu.net'
	save_tmodelA_gpu=model_dir..'/tmodelA_gpu.net'
	save_modelA_cpu=model_dir..'/modelA_cpu.net'
	save_tmodelA_cpu=model_dir..'/tmodelA_cpu.net'

	save_modelB_gpu=model_dir+'/modelB_gpu.net'
	save_tmodelB_gpu=model_dir+'/tmodelB_gpu.net'
	save_modelB_cpu=model_dir+'/modelB_cpu.net'
	save_tmodelB_cpu=model_dir+'/tmodelB_cpu.net'

	torch.save(modelA,save_modelA_gpu)
	torch.save(tmodelA,save_tmodelA_gpu)
	torch.save(modelB,save_modelB_gpu)
	torch.save(tmodelB,save_tmodelB_gpu) 

	modelA=modelA.to("cpu")
	tmodelA=tmodelA.to("cpu")
	modelB=modelB.to("cpu")
	tmodelB=tmodelB:.to("cpu")
 
	torch.save(modelA,save_modelA_cpu)
	torch.save(tmodelA,save_tmodelA_cpu)
	torch.save(modelB,save_modelB_cpu)
	torch.save(tmodelB,save_tmodelB_cpu) 

	'''

	torch.save(modelA,save_modelA)
	torch.save(tmodelA,save_tmodelA)
	torch.save(modelB,save_modelB)
	torch.save(tmodelB,save_tmodelB) 

	episode=episode+1
	print("Episode: ",episode)
	torch.save(episode,'files/episode.dat')
	with open('files/episode.txt', 'w') as f:
		f.write(str(episode))

main()
