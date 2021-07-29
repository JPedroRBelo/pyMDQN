#datageneration
simulation_speed = 5
t_steps = 2000
robot_fov = 60
#environment
raw_frame_height = 320
raw_frame_width = 240
proc_frame_size = 198
state_size = 8
port = 12375        
#host='192.168.0.11'
#host='10.62.6.208'
host='127.0.0.1'
#mdqn
t_episodes=14
#NQL
actions	= ['1','2','3','4']
#epsilon annealing
ep_start   = 1.0
ep_end	 = 0.1
ep_endt_number = 14
ep_endt	= ep_endt_number * t_steps
learn_start= 0
#training
cycles = 10
#trainNQL
device = "cuda"#cuda
t_eps = t_episodes
minibatch_size = 25
discount       = 0.99 #Discount factor.
replay_memory  = 28000
bufferSize     =  t_steps
target_q       = 1
#rewards
neutral_reward = 0
##handshake
hs_success_reward = 1
hs_fail_reward = -0.1

#network
noutputs=4
nfeats=8
nstates=[16,32,64,256]
#kernels = [4,2]
kernels = [9,5]
strides = [3,1]
poolsize=2

