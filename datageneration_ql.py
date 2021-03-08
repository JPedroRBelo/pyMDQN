import torch
import torch.nn as nn
from pathlib import Path

'''
require 'nn'
require 'torch'
require 'robot_environment'
require 'image'
require 'RobotNQL'
require 'os'
'''

t_steps=2050
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(torch.initial_seed())  
win=None

episode=torch.load('files/episode.dat')
#local agent=RobotNQL(episode=episode)
#env=Environment()

dirname_rgb='dataset/RGB/ep'+episode
dirname_dep='dataset/Depth/ep'+episode
dirname_model='results/ep'+episode
Path(dirname_rgb).mkdir(parents=True, exist_ok=True)
Path(dirname_dep).mkdir(parents=True, exist_ok=True)
Path(dirname_model).mkdir(parents=True, exist_ok=True)

episode = int(episode)



def generate_data(episode):
	
	recent_rewards=torch.load('recent_rewards.dat')
	recent_actions=torch.load('recent_actions.dat')
	reward_history=torch.load('files/reward_history.dat')
	action_history=torch.load('files/action_history.dat')
	ep_rewards=torch.load('files/ep_rewards.dat')

	aset = ['1','2','3','4']
	testing = False
	init_step = 0
	
	if(len(reward_history)!=episode):
		if((len(recent_rewards)>0) and (len(recent_rewards)<=t_steps)):
			init_step = len(recent_rewards)
		

	if testing:
		#aset = {'1','1','1','1'}
		aset = ['4','4','4','4']
		init_step = 0


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

	#env.send_data_to_pepper("step"+init_step)
	#env.close_connection()
	#env = Environment()

	reward = 0 #temp
	terminal = 0
	screen = None
	depth = None
	#screen, depth, reward, terminal = env.perform_action('-',init_step+1)

	step=init_step+1
	while step <=t_steps:
		print("Step=",step)
		action_index=0
		numSteps=(episode-1)*t_steps+step
		if reward>15:
			pass
			#action_index = agent.perceive(1, screen,depth, terminal, false, numSteps,step,testing)
		else:
			pass
			#action_index = agent.perceive(0, screen,depth, terminal, false, numSteps,step,testing)
		step=step+1		
		if action_index == None:
				action_index=1
		if not terminal: 
			pass
			#screen,depth,reward,terminal=env.perform_action(aset[action_index],step)
		else:  
			pass
			#screen,depth, reward, terminal = env.perform_action('-',step)

		if step >= t_steps:
			terminal=1

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

	

def main():
	generate_data(episode)
	#env:close_connection()

main()
