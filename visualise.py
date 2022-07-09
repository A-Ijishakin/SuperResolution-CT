import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import nibabel as nib 
import numpy as np 
from Models import createMLP, createCNN, createRN 
from utils import inSizer, t2n, n2t, upsample_image, normalise  
import torch 


##############################

fig, axes = plt.subplots(1, 6)
axes = axes.flatten()

shapes = [(96, 80), (48, 40), (32, 26)] 

#### MLPS
#instantiate 
net = createMLP(inSizer['Four'][0])

#load 
net = torch.load('models/MLP1', map_location=torch.device('cpu'))

#### CNNs 
#instantiate 
cnn = createCNN(inSizer['Four'][1]) 

#load 
cnn = torch.load('models/CNN1', map_location=torch.device('cpu'))

#### RNs 
#instantiate
rn = createRN(inSizer['Four'][2])

#load 
rn = torch.load('models/RN1', map_location=torch.device('cpu'))  

#set an image index 
idx = 627 

#load images 
img1 = nib.load(f'dataset/test/slices/lr/df4/%04d.nii.gz' % (idx+1)).get_fdata() 
img2 = nib.load(f'dataset/test/slices/hr/%04d.nii.gz' % (idx+1)).get_fdata()

#plot the images 
axes[0].imshow(img1, cmap = 'gray') 
axes[0].set_title('low res')
axes[0].axis('off') 
axes[1].imshow(upsample_image(img1.reshape(1, 48, 40)).reshape(192, 160), cmap = 'gray')
axes[1].set_title('Interpolator')
axes[1].axis('off')  
axes[2].imshow(t2n(net(n2t(img1).flatten()).reshape(192, 160)), cmap = 'gray') 
axes[2].set_title('MLP')
axes[2].axis('off') 
axes[3].imshow(t2n(cnn(n2t(img1).unsqueeze(0).unsqueeze(0)).reshape(192, 160)), cmap = 'gray') 
axes[3].set_title('CNN')
axes[3].axis('off') 
axes[4].imshow(t2n(rn(n2t(img1).unsqueeze(0).unsqueeze(0)).reshape(192, 160)), cmap = 'gray')
axes[4].set_title('RES NET') 
axes[4].axis('off')
axes[5].imshow(normalise(img2), cmap = 'gray') 
axes[5].set_title('high res') 
axes[5].axis('off')    

#save the image 
plt.savefig('figure1.png')  

