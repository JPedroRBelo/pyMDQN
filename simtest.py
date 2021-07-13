#!/usr/bin/env python
import signal
import torch
import os
from environment import Environment
import datageneration
import train 
import time
import sys
import subprocess
from subprocess import Popen
from os.path import abspath, dirname, join


def getValue(filename):
	line = subprocess.check_output(['tail', '-1', filename])
	return int(line.decode('utf-8').replace('\n', ''))


def openSim(process):
	process.terminate()
	time.sleep(5)
	process = Popen(command)
	time.sleep(5)
	return process

def killSim(process):
	process.terminate()		
	time.sleep(10)

def signalHandler(sig, frame):
    process.terminate()
    sys.exit(0)


t_episodes=30

file_phase = 'files/phase.txt'

episode=torch.load('files/episode.dat')


command = './simDRLSR.x86_64'
execute_simulator = False
if(len(sys.argv)>1):
	if(sys.argv[1]=='true' or  sys.argv[1]=='True'):
		execute_simulator = True
		if(len(sys.argv)>2):
			directory = str(sys.argv[2])
			command = abspath(join(directory,command))
			#command = directory+command


phase = getValue(file_phase)
process = Popen('false') # something long running
signal.signal(signal.SIGINT, signalHandler)
phase = 0
for i in range(episode,t_episodes):
	#phase = getValue(file_phase)

	if(phase == 0):
		print("Episode: ",i," collection data.")

		if execute_simulator: process = openSim(process)
		recent_rewards=torch.load('recent_rewards.dat')
		reward_history=torch.load('files/reward_history.dat')
		print(len(recent_rewards))
		
		env=Environment(epi=episode)

		env.send_data_to_pepper("start")
		time.sleep(1)
		env.close_connection()
		time.sleep(1)
		#Execute data generation phase script
		#datageneration.main()

		#setValue(file_phase,1)

		env=Environment(epi=episode)
		env.send_data_to_pepper("stop")
		phase = 1



	#phase = getValue(file_phase)
	if(phase == 1):
		print("Episode: ",i," training model.")
		
		print("Sending signal to kill simulator")
		if execute_simulator: killSim(process)
		#setValue('flag_simulator.txt',9)
		time.sleep(10)		
		#Execute train phase script
		#train.main()

		#setValue(file_phase,0)
		phase = 0

	if execute_simulator: killSim(process)
print("Model trained...")

