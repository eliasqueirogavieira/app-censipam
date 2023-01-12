import matplotlib.pyplot as plt
import numpy as np
import glob
from torch import nn
import torch 
import rasterio

from torch.optim import AdamW, SGD
from torch.cuda.amp import GradScaler, autocast

from semseg import schedulers

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from merge_preds import mergeNet, read_sample




def readPreds(in_fname):

    with rasterio.open(in_fname) as src:
        in_preds = src.read()
    
    in_preds = np.expand_dims(in_preds, 0) # [Batch, Preds, H, W]
    in_preds = torch.tensor(in_preds)

    return in_preds



model_fname = '/censipam_data/renam/netMerge/model_merge.pth'

preds_fname = '/censipam_data/renam/netMerge/output/cpred1.tif'







model = mergeNet()
model.load_state_dict(torch.load(model_fname, map_location='cpu'))
model.eval()




print(model)

