import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pathlib import Path
import copy
from TrainNQL import TrainNQL
import os.path
from os import path
import torch.nn as nn
from pathlib import Path
from RobotNQL import RobotNQL
from environment import Environment
import pickle
import time
import shutil
import logging


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

	target_net=cfg.target_q
	agent.load_data()
	for j in range(cycles):
		print("Train= "+str(j+1)+"/"+str(cycles))		
		agent.train()
		if(j%target_net==0):
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




	hspos = 0
	hsneg = 0
	wave = 0
	wait = 0
	look = 0
	t_steps=cfg.t_steps
	dirname_rgb='dataset/RGB/ep'+str(episode)
	dirname_dep='dataset/Depth/ep'+str(episode)
	dirname_model='validation/'+str(episode)
	agent = RobotNQL(epi=episode,cfg=cfg,validation=True)
	env = Environment(cfg)

	Path(dirname_rgb).mkdir(parents=True, exist_ok=True)
	Path(dirname_dep).mkdir(parents=True, exist_ok=True)
	Path(dirname_model).mkdir(parents=True, exist_ok=True)

	env = Environment(cfg)

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

	logger = logging.getLogger()
	logger.setLevel(logging.INFO) # process everything, even if everything isn't printed

	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO) # or any other level
	logger.addHandler(ch)


	fh = logging.FileHandler('validation/'+str(episode)+'/results.log')
	fh.setLevel(logging.INFO) # or any level you want
	logger.addHandler(fh)

	aset = cfg.actions
	testing = -1
	init_step = 0
	
	'''
	if(len(reward_history)!=episode):
		if((len(recent_rewards)>0) and (len(recent_rewards)<=t_steps+1)):
			init_step = len(recent_rewards)
		
	'''

	
	aux_total_rewards = 0
	'''
	for i in range(init_step):
		aux_total_rewards = aux_total_rewards+recent_rewards[i]
	'''

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
	env = Environment(cfg)

	reward = 0 #temp
	terminal = 0
	screen = None
	depth = None
	screen, depth, reward, terminal = env.perform_action('-',init_step+1)

	step=init_step+1
	while step <=t_steps+1:
		print("Step=",step)
		action_index=0
		numSteps=0
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

	

		if action_index == 3 :
			if reward>0 :
				hspos = hspos+1
			elif reward==-0.1 : 
				hsneg = hsneg+1
			
		elif action_index == 0 :
			wait = wait+1
		elif action_index == 1 :
			look = look+1
		elif action_index == 2 :
			wave = wave+1

		

		
	
	
		logger.info('###################')	
		logger.info("STEP:\t"+str(step))
		logger.info('Wait\t'+str(wait))
		logger.info('Look\t'+str(look))
		logger.info('Wave\t'+str(wave))
		logger.info('HS Suc.\t'+str(hspos))
		logger.info('HS Fail\t'+str(hsneg))
		if(hspos+hsneg):
			logger.info('Acuracy\t'+str(((hspos)/(hspos+hsneg))))

		logger.info('================>')
		logger.info("Total Reward: "+str(total_reward))
		logger.info('================>')
		torch.save(rewards,file_recent_rewards)
		torch.save(actions,file_recent_actions)

		
	reward_history.append(rewards)
	action_history.append(actions)
	ep_rewards.append(total_reward)
	print('\n')
	
	 
	torch.save(ep_rewards,file_ep_rewards)
	torch.save(reward_history,file_reward_history)
	torch.save(action_history, file_action_history)

	torch.save([],file_recent_rewards)
	torch.save([],file_recent_actions)






	env.close_connection()



def main(cfg):



	ep_validation = "validation"
	n_validation = 0
	
	while(path.isdir('validation/'+ep_validation+str(n_validation))):
		n_validation+=1 
	
	
	name_ep=ep_validation+str(n_validation)
	print(name_ep)
	Path('validation/'+name_ep).mkdir(parents=True, exist_ok=True)

	
	shutil.copy(cfg.__file__,'validation/'+name_ep+'/')
	#shutil.copy('validation/'+ep_validation+str(n_validation-1)+'/modelDepth.net','validation/'+name_ep+'/')
	#shutil.copy('validation/'+ep_validation+str(n_validation-1)+'/tModelDepth.net','validation/'+name_ep+'/')
	#shutil.copy('validation/'+ep_validation+str(n_validation-1)+'/modelGray.net','validation/'+name_ep+'/')
	#shutil.copy('validation/'+ep_validation+str(n_validation-1)+'/tModelGray.net','validation/'+name_ep+'/')
	#shutil.copy('results/ep60/modelDepth.net','validation/'+name_ep+'/')
	#shutil.copy('results/ep60/tModelDepth.net','validation/'+name_ep+'/')
	#shutil.copy('results/ep60/modelGray.net','validation/'+name_ep+'/')
	#shutil.copy('results/ep60/tModelGray.net','validation/'+name_ep+'/')
	
	train(name_ep,cfg)
	

	#name_ep=ep_validation+str(n_validation-1)
	env=Environment(cfg)

	env.send_data_to_pepper("start")
	time.sleep(1)
	env.close_connection()
	time.sleep(1)
	#Execute data generation phase script
	datavalidation(name_ep,cfg)


	env=Environment(cfg)
	env.send_data_to_pepper("stop")
	
	

if __name__ == "__main__":
	'''
	import validate_config6 as vcfg
	main(vcfg)
	import validate_config7 as vcfg
	main(vcfg)
	import validate_config8 as vcfg
	main(vcfg)
	import validate_config9 as vcfg
	main(vcfg)
	'''
	import configs.validate_config17 as vcfg
	main(vcfg)
	import configs.validate_config18 as vcfg
	main(vcfg)
	import configs.validate_config15 as vcfg
	main(vcfg)
	import configs.validate_config14 as vcfg
	main(vcfg)


