import torch

from gmodel import Net
from pathlib import Path

modelA = Net()
modelB = Net()

save_modelA_gpu='results/ep0/modelA_cuda.net'
save_modelB_gpu='results/ep0/modelB_cuda.net'
save_tmodelA_gpu='results/ep0/tmodelA_cuda.net'
save_tmodelB_gpu='results/ep0/tmodelB_cuda.net'
save_modelA_cpu='results/ep0/modelA_cpu.net'
save_modelB_cpu='results/ep0/modelB_cpu.net'
save_tmodelA_cpu='results/ep0/tmodelA_cpu.net'
save_tmodelB_cpu='results/ep0/tmodelB_cpu.net'
Path('results/ep0').mkdir(parents=True, exist_ok=True)
torch.save(modelA,save_modelA_gpu)
torch.save(modelB,save_modelB_gpu)
torch.save(modelA,save_tmodelA_gpu)
torch.save(modelB,save_tmodelB_gpu)

'''
model=model:float()
tmodel=tmodel:float()

torch.save(model,save_modelA_cpu)
torch.save(model,save_modelB_cpu)
torch.save(model,save_tmodelA_cpu)
torch.save(model,save_tmodelB_cpu	)
'''