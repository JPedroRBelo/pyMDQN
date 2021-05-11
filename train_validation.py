import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pathlib import Path
import copy
from TrainNQL import TrainNQL
import os.path
from os import path
import validate_config as vcfg
import torch.nn as nn
from pathlib import Path
from RobotNQL import RobotNQL
from environment import Environment
import pickle


#device = "cuda"#torch.device("cuda" if torch.cuda.is_available() else "cpu")




#device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(torch.initial_seed())  


#device = "cuda"#torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(episode,cfg):	
	cycles = cfg.cycles
	trains = cfg.trains
	torch.cuda.empty_cache()
	torch.manual_seed(torch.initial_seed()) 
	
	agent=TrainNQL(epi=episode,cfg=cfg,validation=True)

	target_net=4
	agent.load_data()
	for j in range(cycles):
		print("\nTrain= "+str(j+1)+"/"+str(cycles))
		for i in range(trains):
			print("epoch ",i+1)
			agent.train()
		agent.gray_target_net=copy.deepcopy(agent.gray_policy_net)
		agent.depth_target_net=copy.deepcopy(agent.depth_policy_net)	


	gray_policy_net=copy.deepcopy(agent.gray_policy_net)
	depth_policy_net=copy.deepcopy(agent.depth_policy_net)


	gray_target_net = copy.deepcopy(agent.gray_target_net)
	depth_target_net = copy.deepcopy(agent.depth_target_net)

	model_dir='validation/'+str(episode)

	Path(model_dir).mkdir(parents=True, exist_ok=True)

	save_gray_policy_net=model_dir+'/modelGray.net'
	save_gray_target_net=model_dir+'/tModelGray.net'
	save_depth_policy_net=model_dir+'/modelDepth.net'
	save_depth_target_net=model_dir+'/tModelDepth.net'

	torch.save(gray_policy_net,save_gray_policy_net)
	torch.save(gray_target_net,save_gray_target_net)
	torch.save(depth_policy_net,save_depth_policy_net)
	torch.save(depth_target_net,save_depth_target_net) 

def datavalidation(episode,cfg):
	t_steps=cfg.t_steps
	dirname_rgb='dataset/RGB/'+str(episode)
	dirname_dep='dataset/Depth/'+str(episode)
	dirname_model='results/'+str(episode)
	agent = RobotNQL(epi=episode,cfg=cfg,validation=True)
	env = Environment()

	Path(dirname_rgb).mkdir(parents=True, exist_ok=True)
	Path(dirname_dep).mkdir(parents=True, exist_ok=True)
	Path(dirname_model).mkdir(parents=True, exist_ok=True)

	env = Environment()

	file_recent_rewards = 'validation/'+str(episode)+'/recent_rewards.dat'
	file_recent_actions='validation/'+str(episode)+'/recent_actions.dat'
	file_reward_history='validation/'+str(episode)+'/reward_history.dat'
	file_action_history='validation/'+str(episode)+'/action_history.dat'
	file_ep_rewards='validation/'+str(episode)+'/ep_rewards.dat'

	if(not path.exists(file_recent_rewards)):
		torch.save([],file_recent_rewards)
	if(not path.exists(file_recent_actions)):
		torch.save([],file_recent_actions)
	if(not path.exists(file_reward_history)):
		torch.save([],file_reward_history)
	if(not path.exists(file_action_history)):
		torch.save([],file_action_history)
	if(not path.exists(file_ep_rewards)):
		torch.save([],file_ep_rewards)




	recent_rewards=torch.load(file_recent_rewards)
	recent_actions=torch.load(file_recent_actions)
	reward_history=torch.load(file_reward_history)
	action_history=torch.load(file_action_history)
	ep_rewards=torch.load(file_ep_rewards)

	aset = cfg.actions
	testing = False
	init_step = 0
	
	if(len(reward_history)!=episode):
		if((len(recent_rewards)>0) and (len(recent_rewards)<=t_steps+1)):
			init_step = len(recent_rewards)
		
	'''
	if testing:
		#aset = {'1','1','1','1'}
		aset = ['4','4','4','4']
		init_step = 0
	'''

	aux_total_rewards = 0
	for i in range(init_step):
		aux_total_rewards = aux_total_rewards+recent_rewards[i]

	actions = []
	rewards = []

	if(init_step!=0):
		actions= recent_actions
		rewards= recent_rewards

	total_reward = aux_total_rewards
	print(init_step)

	env.send_data_to_pepper("step"+str(init_step))
	env.send_data_to_pepper("episode"+str(episode))
	env.close_connection()
	env = Environment()

	reward = 0 #temp
	terminal = 0
	screen = None
	depth = None
	screen, depth, reward, terminal = env.perform_action('-',init_step+1)

	step=init_step+1
	while step <=t_steps+1:
		print("Step=",step)
		action_index=0
		numSteps=(episode-1)*t_steps+step
		action_index = agent.perceive(screen,depth, terminal, False, numSteps,step,testing)		
		step=step+1		
		if action_index == None:
				action_index=1
		if not terminal: 
			screen,depth,reward,terminal=env.perform_action(aset[action_index],step)
		else:
			screen,depth, reward, terminal = env.perform_action('-',step)

		if step >= t_steps:
			terminal=1

		#handshake reward calc
		if(aset[action_index]=='4'):
			#reward = min(reward,cfg.hs_success_reward)
			#reward = max(reward,cfg.hs_fail_reward)
			if reward>=1:
				reward = cfg.hs_success_reward
			elif reward<0:
				reward = cfg.hs_fail_reward

		rewards.append(reward)
		actions.append(action_index)
		total_reward=total_reward+reward
		print("Total Reward: ",total_reward)
		print('================>')
		torch.save(rewards,recent_rewards)
		torch.save(actions,recent_actions)

		
	reward_history.append(rewards)
	action_history.append(actions)
	ep_rewards.append(total_reward)
	print('+++++++++++++++++++++++++++++++++')
	
	 
	torch.save(ep_rewards,file_ep_rewards)
	torch.save(reward_history,file_reward_history)
	torch.save(action_history, file_action_history)

	torch.save([],file_recent_rewards)
	torch.save([],file_recent_actions)






	env.close_connection()


def main():
	ep_validation = "validation"
	n_validation = 0
	while(path.isdir(ep_validation+str(n_validation))):
		n_validation+=1 
	
	name_ep=ep_validation+str(n_validation)
	Path('validation/'+name_ep).mkdir(parents=True, exist_ok=True)




	train(name_ep,vcfg)

	env=Environment()

	env.send_data_to_pepper("start")
	time.sleep(1)
	env.close_connection()
	time.sleep(1)
	#Execute data generation phase script
	datavalidation(name_ep,vcfg)


	env=Environment()
	env.send_data_to_pepper("stop")

	
	

if __name__ == "__main__":
   main()

