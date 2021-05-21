#datageneration
t_steps = 600
#environment
raw_frame_height = 320
raw_frame_width = 240
proc_frame_size = 84
state_size = 8
port = 12375        
#host='192.168.0.11'
#host='10.62.6.208'
host='127.0.0.1'
#mdqn
t_episodes=60
#NQL
actions	= ['1','2','3','4']
#epsilon annealing
ep_start   = 1.0
ep_end	 = 0.1
ep_endt	= 28000
learn_start= 0
#training
cycles = 1
trains = 10
#trainNQL
device = "cpu"#cuda
t_eps = 60
minibatch_size = 25
discount       = 0.99 #Discount factor.
replay_memory  = 120000
bufferSize     =  2000
target_q       = 4

#rewards
neutral_reward = 0
hs_success_reward = 1
hs_fail_reward = -0.1

#network
noutputs=4
nfeats=8
nstates=[16,32,64,256]
kernels = [4,2]
strides = [3,1]
poolsize=2

