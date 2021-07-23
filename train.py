import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pathlib import Path
import copy
from TrainNQL import TrainNQL
import config as cfg



#device = "cuda"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

cycles = cfg.cycles
#trains = cfg.trains

def main():	
	torch.cuda.empty_cache()
	torch.manual_seed(torch.initial_seed())  
	episode=int(torch.load('files/episode.dat'))

	agent=TrainNQL(epi=episode)

	target_net=cfg.target_q
	agent.load_data()
	for j in range(cycles):
		print("Ep."+str(episode)+" Train= "+str(j+1)+"/"+str(cycles))		
		agent.train()
		'''
		if(j%target_net==0):
			agent.gray_target_net=copy.deepcopy(agent.gray_policy_net)
			agent.depth_target_net=copy.deepcopy(agent.depth_policy_net)	
		'''


	gray_policy_net=copy.deepcopy(agent.gray_policy_net)
	depth_policy_net=copy.deepcopy(agent.depth_policy_net)

	'''
	if episode%target_net==1:
		agent.gray_target_net=copy.deepcopy(agent.gray_policy_net)
		agent.depth_target_net=copy.deepcopy(agent.depth_policy_net)
	'''

	gray_target_net = copy.deepcopy(agent.gray_target_net)
	depth_target_net = copy.deepcopy(agent.depth_target_net)

	model_dir='results/ep'+str(episode)

	Path(model_dir).mkdir(parents=True, exist_ok=True)

	save_gray_policy_net=model_dir+'/modelGray.net'
	save_gray_target_net=model_dir+'/tModelGray.net'
	save_depth_policy_net=model_dir+'/modelDepth.net'
	save_depth_target_net=model_dir+'/tModelDepth.net'

	torch.save(gray_policy_net,save_gray_policy_net)
	torch.save(gray_target_net,save_gray_target_net)
	torch.save(depth_policy_net,save_depth_policy_net)
	torch.save(depth_target_net,save_depth_target_net) 

	episode=episode+1
	print("Episode: ",episode)
	torch.save(episode,'files/episode.dat')
	with open('files/episode.txt', 'w') as f:
		f.write(str(episode))	

if __name__ == "__main__":
   main()
