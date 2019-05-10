import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import utils


class TemporalVAE(nn.Module):

    def __init__(self):
        super(TemporalVAE, self).__init__()

        
