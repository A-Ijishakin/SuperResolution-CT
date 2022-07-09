import torch 
import torch.nn as nn 
import numpy as np 
import tensorflow as tf 
import nibabel as nib 
from glob import glob 
import matplotlib.pyplot as plt 
import os 
from tensorflow.keras.models import load_model


def weight_initialisation(model):
  """ weight initialisation  
  
  A function which conducts xavier initialisation on the weights of a convolutional neural network created in PyTorch. 

  Args: 
    - model: The model which will be operated upon. 

  Output:
    - model: The model following xavier initialisation. 
  """
  if isinstance(model, nn.Linear):
      nn.init.xavier_uniform(model.weight)
      model.bias.data.fill_(0.01) 

def weights_init(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0) 

def weight_init(model):
  """ weight init  
  
  A function which initialises (PyTorch) model weights in-order to have mean zero and a standard deviation of 0.01. 

  Args: 
    - model: The model which will be operated upon. 

  Output:
    - model: The model following weight initialisation. 
  """
  if type(model) == nn.Linear:
      nn.init.normal_(model.weight, std=0.01)

def t2n(tensor):
  """ t2n 
  A function which transforms a PyTorch tensor into a numpy.ndarray. 

  Args: 
    - tensor: The torch.Tensor which will be transformed. 

  Output:
    - A numpy.ndarray which was formerly a torch.Tensor 
  
  """
  return tensor.cpu().detach().numpy()  

def n2t(ndarray, device=None):
  """ t2n 
  A function which transforms a numpy.ndarray into a PyTorch tensor. 

  Args: 
    - ndarray: A numpy.ndarray which will be transformed. 
    

  Output:
    - tensor: The torch.Tensor which was formerly an numpy.ndarray. 
  
  """
  tensor = torch.Tensor(ndarray) 
  if device == 'cuda':
    tensor = tensor.cuda()

  return tensor  

def normalise(data): 
  """ normalise 
  
  A function which normlaises an image (stored as a numpy.ndarray) to a have values between 0 and 1. 

  Args:
    - data: the image that will be normalised.

  Output:
    - a normalised image. 
  
  """
  return (data - data.min())/(data.max() - data.min())

def downsample_image(img, n=2):
  """ downsample_image 
  
  A function which downsamples an image (stored as a numpy.ndarray). 

  Args: 
    - img: The image which will be downsampled 
    - n: The factor by which the image should be downsampled 

  Output:
    - A downsampled image. 
  """
  shape = np.array(img.shape)
  img = img[tf.newaxis, ... , tf.newaxis]
  img_slice_downsampled = tf.image.resize(img, shape/n, method = 'bicubic' , preserve_aspect_ratio=False, antialias=False).numpy()
  
  img_slice_downsampled = img_slice_downsampled[0,:,:,0]      
  
  return img_slice_downsampled 


def upsample_image(img):
  """ upsample_image 
  
  A function which upsamples an image (stored as a numpy.ndarray). To the original high resolution (192 * 160), size. 

  Args: 
    - img: The image which will be upsampled.

  Output:
    - An upsampled image. 
  """
  
  img = img[..., tf.newaxis]
  
  img_slice_upsampled = tf.image.resize(img, [192, 160], method = 'bicubic' , preserve_aspect_ratio=False, antialias=False).numpy()
  
  img_slice_upsampled = img_slice_upsampled[0,:,:,0] 
  img_slice_upsampled = img_slice_upsampled.squeeze()
       
  
  return img_slice_upsampled 

def createArchitecture(out_channels):
  """
  createArchitecture 

  A function which creates a list outlining a CNN architecture 

  Args: 
    out_channels - The architecture. It must be a list of lists, where the first 
                   entry is the number of output channels and the second is how many there are. 

  Output:
    Architecture - A list with the correct model architecture. 

  """
  Architecture = np.array([], dtype = np.int8)
  
  for layer in out_channels:   
    Architecture = np.concatenate((Architecture, np.array([layer[0] * int(x) for x in np.ones(layer[1])]))) 
    
  Architecture = [int(x) if x.isalpha() != True else x for x in Architecture]       
            
  return Architecture  


def PSNR(high_res, prediction):
  """ PSNR
  
  A function which calculates the peak signal to noise ratio between two images. 

  Args:
    - high_res: The original high resolution image (numpy.ndarray). 
    - prediction: The upsampled low res image (numpy.ndarray). 

  Output:
    - psnr: The peak signal to noise ratio between the two images. 
  """

  mean_squaredError = np.mean((high_res - prediction) ** 2)
  if(mean_squaredError == 0):             
      return 100
      
  max_val = 1
  psnr = 20 * np.log10(max_val / (mean_squaredError ** 0.5))
  return psnr  

def ssim(high_res, prediction):
  """ ssim 
  
  A function which calculates the peak structural similarity index measure between two images. 

  Args:
    - high_res: The original high resolution image (numpy.ndarray). 
    - prediction: The upsampled low res image (numpy.ndarray). 

  Output:
    - : Structural similarity index measure between the two images. 
  """
  prediction = prediction[tf.newaxis, ..., tf.newaxis]

  high_res = high_res[...,  tf.newaxis]
 
  return tf.image.ssim(high_res, prediction, 1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03) 


#make note of the relevant input sizes for the different models given the downsampling size, the first indices of the corresonding arrays
#is: MLPs, CNNs, RNs 
inSizer = {'Two':[(96*80), 20480, 2048], 'Four':[(48*40), 3072, 2048], 'Six':[(32*26), 512, 2048]} 


  