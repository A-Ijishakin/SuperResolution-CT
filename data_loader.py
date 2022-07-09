import torch 
import os 
from glob import glob 
from utils import normalise 
import nibabel as nib 

""" Written by Ahmed Abdulaal of University College London """

#initialise batch_size 
batch_size = 32  
 
#set the size of the dataset appropriately 
N = 5181 

# load the slices
class SliceLoader(torch.utils.data.Dataset):
    """ SliceLoader 
    
    A class which is used to allow for efficient data loading of the training data. 

    Args:
        - torch.utils.data.Dataset: A PyTorch module from which this class inherits which allows it 
        to make use of the PyTorch dataloader functionalities. 
    
    """
    def __init__(self, downsampling_factor, dtype= 'train', N=N, folder_name=None, is_train=True):

        """ Class constructor
        
        Args:
            - downsampling_factor: The factor by which the loaded data has been downsampled. 
            - N: The length of the dataset. 
            - folder_name: The folder from which the data comes from 
            - is_train: Whether or not the dataloader is loading training data (and therefore randomised data).   
        """
        #set the folder name is needed. 
        if folder_name: 
          self.folder_name = folder_name 
        
        #set the training status 
        self.is_train = is_train
        #set the downsampling factor 
        self.downsampling_factor = downsampling_factor 
        
        self.dtype = dtype 
        self.N = N
 
    def __len__(self):
        """ __len__
        
        A function which configures and returns the size of the datset. 
        
        Output: 
            - N: The size of the dataset. 
        """
        return (self.N)

    def __getitem__(self, idx):
        """ __getitem__
        
        A function which loads and returns the low resolution image and it's label. 
        
        Args:
            - idx: The index of the low resolution image and its label. 

        Output:
            - image: A low resolution image from the training set. 
            - label: It's high resolution label. 
        """

        
        #load in the image and its label
        image = self._load_nib(f'dataset/{self.dtype}/slices/lr/df{self.downsampling_factor}/%04d.nii.gz' % (idx + 1))
        label = self._load_nib(f'dataset/{self.dtype}/slices/hr/%04d.nii.gz' % (idx + 1))
        return image, label 

    def _load_nib(self, filename): 
        """ _load_nib 
        
        A function to load compressed nifti images.

        Args:
            - filename: The name of the file to be loaded. 

        Ouput:
            - The corresponding image as a PyTorch tensor. 
        
        """
        return torch.tensor(normalise(nib.load(filename).get_fdata()), dtype=torch.float)

