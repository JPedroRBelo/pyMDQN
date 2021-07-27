import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import configValidation as cfg
import sys
import os.path
from os import path



folder = 'files'



t_steps=cfg.t_steps
datContent = []





v_hspos = []
v_hsneg = []
v_wave  = []
v_wait  = []
v_look  = []
v_hsacc = []
v_wvacc = []
v_rewards = []



validation_dirs = 'validation'

epochs = 0

while(path.isdir(validation_dirs+str(epochs))):
		epochs+=1

epochs = epochs-1
print(epochs)



for i in range(epochs+1):

	hspos = 0
	hsneg = 0
	wave = 0

	wait = 0
	look = 0
	rwrd = 0
	folder = validation_dirs+str(i)

	rewards_file = folder+'/reward_history.dat'


	filename = rewards_file

	rewards=torch.load(folder+'/reward_history.dat')#.detach().cpu().numpy()
	actions=torch.load(folder+'/action_history.dat')#.detach().cpu().numpy()
	flag = False

	for step in range(t_steps):	

		if(len(actions)>0 ):
			flag = True
			rwrd = rwrd+rewards[0][step]
			if actions[0][step] == 3 :
				if rewards[0][step]>0 :
					hspos = hspos+1
				elif rewards[0][step]==cfg.hs_fail_reward : 
					hsneg = hsneg+1
			
			elif actions[0][step] == 0 :
				wait = wait+1
			elif actions[0][step] == 1 :
				look = look+1
			elif actions[0][step] == 2 :
				wave = wave+1
				
		
	
	if(flag):
		print('###################')
		print('Epoch\t',i+1)	
		print('Wait\t',wait)
		print('Look\t',look)
		print('Wave\t',wave)
		print('HS Suc.\t',hspos)
		print('HS Fail\t',hsneg)

		
		ha = 0
		if(hspos + hsneg != 0):		
			ha = ((hspos)/(hspos+hsneg))
		print('HS Acuracy\t',ha)	
		print('Cumulative Reward\t',rwrd)
		
		v_wait.append(wait)
		v_look.append(look)
		v_wave.append(wave)

		v_hspos.append(hspos)
		v_hsneg.append(hsneg)
		v_hsacc.append(ha)
		v_rewards.append(rwrd)


clip = len(v_hsacc)


arg = 'all'

norm = (v_rewards - np.min(v_rewards))/np.ptp(v_rewards)



if len(sys.argv) > 1:
	arg = sys.argv[1]
if len(sys.argv) > 2:
	clip = int(sys.argv[2])

clip_max = min(clip,len(v_hsacc))

if (arg == 'rwds'):
	#plt.ylim([0, 1])
	plt.ylabel('Cumulative Reward')
	
	plt.plot(v_rewards,label='Reward')
	#plt.plot(np.array(v_hsacc)*1000,label='Handshake')
	plt.grid()
elif (arg == 'norm'):
	#plt.ylim([0, 1])
	plt.ylabel('')
	plt.xlabel('Epoch ')
	
	plt.plot(v_hsacc[:clip_max],label='Handshake Acuracy (%)')
	plt.plot(norm[:clip_max],label='Nomalized Cumulative Reward')
	#plt.plot(np.array(v_hsacc)*1000,label='Handshake')
	#plt.set_xlim(0, clip_max)
	dim=np.arange(0,clip_max,1)
	dimy=np.arange(0.0,1.1,0.1)
	plt.xticks(dim)
	plt.yticks(dimy)
	plt.grid()
elif (arg == 'both'):
	t = np.arange(0.0, clip_max, 1)

	fig, ax1 = plt.subplots()
	plt.grid()
	color = 'tab:blue'
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Acuracy', color=color)
	ax1.plot(t, v_hsacc[:clip_max], color=color)
	dim=np.arange(0,clip_max,1)
	ax1.set_xticks(dim)
	dimy=np.arange(0.0,1.1,0.1)
	ax1.set_yticks(dimy)
	dimy=np.arange(0.0,1.1,0.1)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:orange'

	ax2.set_ylabel('Cumulative Reward', color=color)  # we already handled the x-label with ax1
	ax2.plot(t, v_rewards[:clip_max], color=color)
	dimy=np.arange(-60,170,20)

	ax2.set_yticks(dimy)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	

else:
	plt.ylim([0, t_steps])
	plt.ylabel('Number of Actions')
	plt.xlabel('Epoch')

	if (arg == 'hs') or (arg == 'all') or (arg == 'reward'):
		plt.plot(v_hspos,label='HS Success')
		plt.plot(v_hsneg,label='HS Fail')

		
	if (arg == 'other') or (arg == 'all'):
		plt.plot(np.add(v_hspos, v_hsneg),label='HS Total')
		plt.plot(v_wait,label='Wait')
		plt.plot(v_look,label='Look')
		plt.plot(v_wave,label='Wave')

	if (arg == 'acc'):
		plt.ylim([0, 1])
		plt.ylabel('Acuracy (%)')
		plt.plot(v_hsacc,label='Handshake')
		plt.plot(v_wvacc,label='Wave')






plt.legend()
plt.show()



