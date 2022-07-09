from data import download_data
download_data() 
import organise_data     
import torch 
import torch.nn as nn 
import numpy as np 
import nibabel as nib 
from Models import trainRNs, trainCNNs, trainMLPs, trainGAN
from utils import inSizer


#hyper-parameter initialisation
batch_size = 32
lr, num_epochs = 1e-3, 15 
loss = nn.MSELoss(reduction='mean')  

#train the MLPs 
trainMLPs([inSizer['Two'][0], inSizer['Four'][0], inSizer['Six'][0]], [2, 4, 6], lr, num_epochs, batch_size, loss) 

#train the CNNs
trainCNNs([inSizer['Two'][1], inSizer['Four'][1], inSizer['Six'][1]], [2, 4, 6], lr, num_epochs, batch_size, loss)

#train the res-nets 
trainRNs([inSizer['Two'][2], inSizer['Four'][2], inSizer['Six'][2]], [2, 4, 6], lr, num_epochs, batch_size, loss)  



  
