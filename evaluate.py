import torch 
import numpy as np 
import os 
from tqdm import tqdm
from glob import glob 
from utils import upsample_image, PSNR, ssim, n2t, t2n, inSizer
from Models import createCNN, createMLP, createRN 
from data_loader import SliceLoader

#set the batch size to 1 so that test images are passed through individually 
batch_size = 1

#set the total number of images to the amount of test images 
N = len(glob(f'dataset/test/slices/lr/df4/*')) 

#set the downsampling factors 
downsampling_factors = [2, 4, 6] 
  
#set the model name appropriately 
model_names = ['MLP', 'CNN', 'RN', 'BI']

#create dictionaries to store the snr and ssim scores 
snr_scores, ssim_scores = [{f'{model}{num}':[] for model in model_names} for num in range(3)], [{f'{model}{num}':[] for model in model_names} for num in range(3)]
snr_scores, ssim_scores  = {key: value for Dict in snr_scores for key, value in Dict.items()}, {key: value for Dict in ssim_scores for key, value in Dict.items()}  

snr_avg, ssim_avg = snr_scores, ssim_scores 
snr_std, ssim_std = snr_scores, ssim_scores 


for idx, (iS, downsampling_factor) in enumerate(zip(list(inSizer.keys()), downsampling_factors)):
    #create the model
    mlp = createMLP(inSizer[iS][0]) 
    cnn = createCNN(inSizer[iS][1])
    rn = createRN(inSizer[iS][2]) 
    
    #load the weights 
    mlp = torch.load(f'models/MLP{idx}')
    cnn = torch.load(f'models/CNN{idx}')
    rn = torch.load(f'models/RN{idx}') 

    #instiate test set
    test_set = SliceLoader(downsampling_factor, dtype= 'test', N=N, is_train=False)
    
    #instatiate data loader 
    test_loader = torch.utils.data.DataLoader(
    test_set, 
    batch_size=batch_size, 
    shuffle=False)  

    
    for img, label in test_loader:  
        #predict to produce hr images 
        output1 = t2n(mlp(n2t(img.flatten())).reshape(192, 160)) 
        output2 = t2n(cnn(n2t(img).unsqueeze(0)).reshape(192, 160))
        output3 = t2n(rn(n2t(img).unsqueeze(0)).reshape(192, 160)) 
        output4 = upsample_image(img).reshape(192, 160) 

        #calculate psnr
        SNR1, SNR2 = PSNR(t2n(label), output1), PSNR(t2n(label), output2)
        SNR3, SNR4 = PSNR(t2n(label), output3), PSNR(t2n(label), output4)
        
        #calculate ssim 
        SSIM1, SSIM2 = ssim(label, output1), ssim(label, output2) 
        SSIM3, SSIM4 = ssim(label, output3), ssim(label, output4)

        #store the snr scores appropriately 
        snr_scores[f'MLP{idx}'].append(SNR1), snr_scores[f'CNN{idx}'].append(SNR2)
        snr_scores[f'RN{idx}'].append(SNR3), snr_scores[f'BI{idx}'].append(SNR4) 

        #store the ssim scores appropriately 
        ssim_scores[f'MLP{idx}'].append(SNR1), ssim_scores[f'CNN{idx}'].append(SNR2)
        ssim_scores[f'RN{idx}'].append(SNR3), ssim_scores[f'BI{idx}'].append(SNR4) 


    #calculate and store the average snr 
    snr_avg[f'MLP{idx}'], snr_avg[f'CNN{idx}'] = np.mean(snr_scores[f'MLP{idx}']), np.mean(snr_scores[f'CNN{idx}']) 
    snr_avg[f'RN{idx}'], snr_avg[f'BI{idx}'] = np.mean(snr_scores[f'RN{idx}']), np.mean(snr_scores[f'BI{idx}']) 
    #calculate and store the average ssim 
    ssim_avg[f'MLP{idx}'], ssim_avg[f'CNN{idx}'] = np.mean(ssim_scores[f'MLP{idx}']), np.mean(ssim_scores[f'CNN{idx}']) 
    ssim_avg[f'RN{idx}'], ssim_avg[f'BI{idx}'] = np.mean(ssim_scores[f'RN{idx}']), np.mean(ssim_scores[f'BI{idx}']) 
    #calculate and store the standard deviation of snr scores  
    snr_std[f'MLP{idx}'], snr_std[f'CNN{idx}'] = np.std(snr_scores[f'MLP{idx}']), np.std(snr_scores[f'CNN{idx}']) 
    snr_std[f'RN{idx}'], snr_std[f'BI{idx}'] = np.std(snr_scores[f'RN{idx}']), np.std(snr_scores[f'BI{idx}']) 
    #calculate and store the standard deviation of ssim scores
    ssim_std[f'MLP{idx}'], ssim_std[f'CNN{idx}'] = np.std(ssim_scores[f'MLP{idx}']), np.std(ssim_scores[f'CNN{idx}']) 
    ssim_std[f'RN{idx}'], ssim_std[f'BI{idx}'] = np.std(ssim_scores[f'RN{idx}']), np.std(ssim_scores[f'BI{idx}']) 



    







