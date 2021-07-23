import torch

from network import DQN
from pathlib import Path
import config as cfg


modelGray = DQN(noutputs=cfg.noutputs,nfeats=cfg.nfeats,nstates=cfg.nstates,kernels=cfg.kernels,strides=cfg.strides,poolsize=cfg.poolsize)

save_modelGray='results/ep0/modelGray.net'
save_tModelGray='results/ep0/tModelGray.net'
Path('results/ep0').mkdir(parents=True, exist_ok=True)
torch.save(modelGray,save_modelGray)
torch.save(modelGray,save_tModelGray)

