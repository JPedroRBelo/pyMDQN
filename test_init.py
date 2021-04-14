import torch

from network import DQN
from pathlib import Path

modelGray = DQN()
modelDepth = DQN()

save_modelGray='results/ep0/modelGray.net'
save_modelDepth='results/ep0/modelDepth.net'
save_tModelGray='results/ep0/tModelGray.net'
save_tModelDepth='results/ep0/tModelDepth.net'
Path('results/ep0').mkdir(parents=True, exist_ok=True)
torch.save(modelGray,save_modelGray)
torch.save(modelDepth,save_modelDepth)
torch.save(modelGray,save_tModelGray)
torch.save(modelDepth,save_tModelDepth)

