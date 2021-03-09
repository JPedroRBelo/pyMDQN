import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import socket
import time
from PIL import Image


class Environment:
	def __init__(self):
		# if gpu is to be used
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		#self.r_len=8
		self.raw_frame_height= 320
		self.raw_frame_width=  240
		self.proc_frame_size=84
		self.state_size=8
		self.frame_per_sec=1
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		port = 12375        
		host='192.168.0.11'
		#host='10.62.6.208'
		host='127.0.0.1'
		flag_connection = False
		
		while(not flag_connection):
			try:
				self.client =self.socket.connect((host, port))
				flag_connection = True
			except socket.error:
				print("Can't connect with robot! Trying again...")
				with open('flag_simulator.txt', 'w') as f:
					f.write(str(1))	
				time.sleep(1)
		with open('flag_simulator.txt', 'w') as f:
				f.write(str(0))		

	def get_tensor_from_image(self,file):
		convert = T.Compose([T.ToPILImage(),
			T.Resize((self.proc_frame_size,self.proc_frame_size), interpolation=Image.BILINEAR),
			T.ToTensor()])
		screen = Image.open(file)
		screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
		screen = torch.from_numpy(screen)
		screen = convert(screen).to(self.device)
		return screen

	def pre_process(self,step):	
		print('Preprocessing images')
		proc_image=torch.cuda.FloatTensor(self.state_size,self.proc_frame_size,self.proc_frame_size)
		proc_depth=torch.cuda.FloatTensor(self.state_size,self.proc_frame_size,self.proc_frame_size)
		episode=torch.load('files/episode.dat')
		dirname_rgb='dataset/RGB/ep'+episode
		dirname_dep='dataset/Depth/ep'+episode
		for i in range(1,self.state_size+1):

			grayfile=dirname_rgb+'/image_'+str(step)+'_'+str(i)+'.png'
			depthfile=dirname_dep+'/depth_'+str(step)+'_'+str(i)+'.png'
			grayimage = Image.open(grayfile)
			proc_image[i-1] = self.get_tensor_from_image(grayfile)
			proc_depth[i-1] = self.get_tensor_from_image(depthfile)			

		return proc_image,proc_depth

	
	def send_data_to_pepper(self,data):
		print('Send data connected to Pepper')
		self.socket.send(data.encode())
		print('Sending data to Pepper')
		while True:
			data = self.socket.recv(1024).decode()
			if data:
				return float(data.replace(',','.'))
			break
		print("Connected with the server")
		return 0

	def perform_action(self,action,step):
		r=self.send_data_to_pepper(action)
		s,d=self.pre_process(step)
		term = False
		return s,d,r,term
	
	def close_connection(self):
		self.socket.close()

