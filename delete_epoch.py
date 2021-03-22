import torch
import numpy as np
import pickle
import sys




	


datContent = []


def remove(row):
	folder = 'files'
	rewards=torch.load(folder+'/reward_history.dat')#.detach().cpu().numpy()
	actions=torch.load(folder+'/action_history.dat')#.detach().cpu().numpy()

	#torch.save(rewards[row],folder+'/bkup_rewards.dat'+str(row))
	#torch.save(actions[row],folder+'/bkup_actions.dat'+str(row))
	print(len(rewards))
	print(len(actions))

	new_rewards = np.delete(rewards, row, 0)
	new_actions = np.delete(actions, row, 0)

	#torch.save(new_rewards,folder+'/reward_history.dat')
	#torch.save(new_actions,folder+'/action_history.dat')




if len(sys.argv) > 1:
	row = int(sys.argv[1])
	print('Removing row: ',row)
	remove(row)

	


