import torch
import pickle

e=[]
episode=1
#datagenaration phase = 0
#train phase = 1
with open('files/phase.txt', 'w') as f:
	f.write(str(0))

torch.save(e,'recent_rewards.dat')
torch.save(e,'recent_actions.dat')
torch.save(e,'files/reward_history.dat')
torch.save(e,'files/action_history.dat')
torch.save(e,'files/ep_rewards.dat')  
torch.save(episode,'files/episode.dat')
with open('files/episode.txt', 'w') as f:
	f.write(str(episode))

#torch.save(e,'files/q1_max_s_ep.dat')
#torch.save(e,'files/q1_max_d_ep.dat')
#torch.save(e,'files/q_max_s_ep.dat')
#torch.save(e,'files/q_max_d_ep.dat')
#torch.save(e,'files/td_err_s_ep.dat')
#torch.save(e,'files/td_err_d_ep.dat')