import torch
import numpy as np
import pickle


folder = 'files'



t_steps=2000
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
print(actions)

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


for i in range(len(actions)):
	hspos = 0
	hsneg = 0
	wave = 0
	wait = 0
	look = 0

	for step in range(t_steps):		

		if actions[i][step] == 3 :
			if rewards[i][step]>0 :
				hspos = hspos+1
			elif rewards[i][step]==-0.1 : 
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
	print('Acuracy\t',((hspos)/(hspos+hsneg)))	



