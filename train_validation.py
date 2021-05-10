import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pathlib import Path
import copy
from TrainNQL import TrainNQL
import os.path
from os import path
import config as cfg
import torch.nn as nn
from pathlib import Path
from RobotNQL import RobotNQL
from environment import Environment


#device = "cuda"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

cycles = cfg.cycles
trains = cfg.trains

t_steps=cfg.t_steps
#device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(torch.initial_seed())  


#device = "cuda"#torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():	
	torch.cuda.empty_cache(episode)
	torch.manual_seed(torch.initial_seed()) 
	
	agent=TrainNQL(epi=episode,validation=True)

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

def datavalidation(episode):

	dirname_rgb='dataset/RGB/'+str(episode)
	dirname_dep='dataset/Depth/'+str(episode)
	dirname_model='results/'+str(episode)
	agent = RobotNQL(epi=episode,validation=True)
	env = Environment()

	Path(dirname_rgb).mkdir(parents=True, exist_ok=True)
	Path(dirname_dep).mkdir(parents=True, exist_ok=True)
	Path(dirname_model).mkdir(parents=True, exist_ok=True)

	env = Environment()
	recent_rewards=torch.load('validation/'+str(episode)+'/recent_rewards.dat')
	recent_actions=torch.load('validation/'+str(episode)+'/recent_actions.dat')
	reward_history=torch.load('validation/'+str(episode)+'/reward_history.dat')
	action_history=torch.load('validation/'+str(episode)+'/action_history.dat')
	ep_rewards=torch.load('validation/'+str(episode)+'/ep_rewards.dat')

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
		torch.save(rewards,'recent_rewards.dat',)
		torch.save(actions,'recent_actions.dat')

		
	reward_history.append(rewards)
	action_history.append(actions)
	ep_rewards.append(total_reward)
	print('+++++++++++++++++++++++++++++++++')
	
	 
	torch.save(ep_rewards,'files/ep_rewards.dat')
	torch.save(reward_history,'files/reward_history.dat')
	torch.save(action_history,'files/action_history.dat')

	torch.save([],'recent_rewards.dat')
	torch.save([],'recent_actions.dat')






	env.close_connection()


def main():
	ep_validation = "validation"
	n_validation = 0
	while(path.isdir(ep_validation+str(n_validation))):
		n_validation+=1 
	
	name_ep=ep_validation+str(n_validation)
	train(name_ep)
	datavalidation(name_ep)

if __name__ == "__main__":
   main()

