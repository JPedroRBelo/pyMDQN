import torch
import torch.optim as optim
import numpy as np
import os
from gmodel import Net

class TransitionTable:
    def __init__(self,stateDim, numActions,histLen, gpu, maxSize, histType,histSpacing,bufferSize):
        self.stateDim = stateDim
        self.numActions = numActions or 4
        self.histLen = histLen or 8
        self.maxSize = maxSize or 30000  # replay memory_size
        self.bufferSize = bufferSize or 3000
        self.histType = histType or "linear"
        self.histSpacing = histSpacing or 1
        self.gpu = gpu
        self.buf_ind = 1
        self.batch_ind_y = 1
        self.batch_ind_d = 1
        self.histIndices = []
        self.numEntries = 0
        self.insertIndex = 0