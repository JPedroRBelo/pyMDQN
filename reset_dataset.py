import os

confirmation = 'no'

print('Are you sure you want to clear all training data and image base and reset all variables to zero state?')

confirmation = input("Type 'yes' to confirm: ")
if(confirmation=='yes'):

	os.system("rm -rf dataset/RGB/ep*")
	os.system("rm -rf dataset/RGB/ep*")
	os.system("rm -rf results/*")
	os.system("python makefiles.py")
	os.system("python init_model.py")
	#os.system("mkdir results/ep0")
	#os.system("cp -RT ep13/ results/ep0")
	os.system("python makefiles.py")
	#os.system("python mdqn.py")
	print('Database reseted!')
else:
	print('Database reset canceled!')