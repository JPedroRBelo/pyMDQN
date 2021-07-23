import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import config as cfg
import sys



folder = 'files'



t_steps=cfg.t_steps
datContent = []



rewards_file = folder+'/reward_history.dat'

#rewards = pickle.load(rewards_file,encoding='latin-1')

'''
table = []
with open(rewards_file, "rb") as fin:
  	buf = fin.readlines()
  	for i in buf:
  		bytess = map(int, i)
  		l = list(bytess)
  		table.append(l)
'''

filename = rewards_file

rewards=torch.load(folder+'/reward_history.dat')#.detach().cpu().numpy()
actions=torch.load(folder+'/action_history.dat')#.detach().cpu().numpy()


'''
objects = []
with (open(rewards_file,encoding='latin-1')) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
'''




#rewards=torch.load(folder+'/reward_history.dat')
#actions=torch.load(folder+'/action_history.dat')
#datContent = open(rewards_file,encoding='latin-1')
#datContent=torch.load(rewards_file,encoding='bytes')
#datContent =byte_array.decode(datContent)
'''
with open(rewards_file,encoding='latin-1') as f:
	datContent = list(f)
	#datContent = list(f.decode('utf8'))

print(datContent)
'''
#results = {{}}

v_hspos = []
v_hsneg = []
v_wave  = []
v_wait  = []
v_look  = []
v_hsacc = []
v_wvacc = []
v_rewards = []

epochs = len(actions)


if len(sys.argv) > 2:
	epochs = int(sys.argv[2])



for i in range(epochs):
	hspos = 0
	hsneg = 0
	wave = 0

	wait = 0
	look = 0
	rwrd = 0

	for step in range(t_steps):		
		if(len(actions[i])>0 ):
			rwrd = rwrd+rewards[i][step]
			if actions[i][step] == 3 :
				if rewards[i][step]>0 :
					hspos = hspos+1
				elif rewards[i][step]==cfg.hs_fail_reward : 
					hsneg = hsneg+1
			
			elif actions[i][step] == 0 :
				wait = wait+1
			elif actions[i][step] == 1 :
				look = look+1
			elif actions[i][step] == 2 :
				wave = wave+1
				
		
	
	
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





arg = 'all'




if len(sys.argv) > 1:
	arg = sys.argv[1]

if (arg == 'rwds'):
	#plt.ylim([0, 1])
	plt.ylabel('Cumulative Reward')
	
	plt.plot(v_rewards,label='Reward')
	#plt.plot(np.array(v_hsacc)*1000,label='Handshake')
	plt.grid()
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



