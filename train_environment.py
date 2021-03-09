import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import socket
import time
from PIL import Image

r_len=8 #--recording time in sec
raw_frame_height=320 #height and width of captured frame
raw_frame_width=240 #height and width of captured frame
proc_frame_size=84 #
state_size=8
frame_per_sec=1
step=1

def get_tensor_from_image(file):
		convert = T.Compose([T.ToPILImage(),
			T.Resize((proc_frame_size,proc_frame_size), interpolation=Image.BILINEAR),
			T.ToTensor()])
		screen = Image.open(file)
		screen = np.ascontiguousarray(screen, dtype=np.float32)/255
		screen = torch.from_numpy(screen)
		screen = convert(screen)
		return screen

def get_data(episode,tsteps):
	images=torch.FloatTensor(tsteps,state_size,proc_frame_size,proc_frame_size)
	depths=torch.FloatTensor(tsteps,state_size,proc_frame_size,proc_frame_size)
	dirname_rgb='dataset/RGB/ep'+str(episode)
	dirname_dep='dataset/Depth/ep'+str(episode)
	for step in range(tsteps):
		proc_image=torch.FloatTensor(state_size,proc_frame_size,proc_frame_size)
		proc_depth=torch.FloatTensor(state_size,proc_frame_size,proc_frame_size)
		dirname_rgb='dataset/RGB/ep'+str(episode)
		dirname_dep='dataset/Depth/ep'+str(episode)
		for i in range(state_size):

			grayfile=dirname_rgb+'/image_'+str(step+1)+'_'+str(i+1)+'.png'
			depthfile=dirname_dep+'/depth_'+str(step+1)+'_'+str(i+1)+'.png'
			proc_image[i] = get_tensor_from_image(grayfile)
			proc_depth[i] = get_tensor_from_image(depthfile)			

		
		images[step]=proc_image.unsqueeze(0)
		depths[step]=proc_depth.unsqueeze(0)	
	return images,depths	


images,depths = get_data(1,5)
print(images.shape)