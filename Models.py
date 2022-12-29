import torch 
import os 
import nibabel as nib 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d 
import torch.optim as optim     
from utils import weight_initialisation, weight_init, weights_init, createArchitecture
from data_loader import SliceLoader
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import torchvision.utils as vutils 
import time  

#create a folder to store model weights if it does not already exist 
if os.path.exists('models') == False:
  os.makedirs('models') 

#if cuda is available then use it 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#############  MULTILAYER PERCEPTRONS ############# 
def createMLP(in_size): 
  """ createMLP

  A function which creates a multi-layer perceptron 
  
  """
  net = nn.Sequential(nn.Linear(in_size, 256),
                  nn.ReLU(),
                  nn.Dropout(0.1),
                  nn.Linear(256, 192*160),
                  nn.Sigmoid()) 
  return net 

def trainMLPs(input_sizes, downsampling_factors, lr, num_epochs, batch_size, loss):
  """ trainMLPS
  
  A function which trains various MLPs based on the downsampling factors. 

  Args:
    - input_sizes: The input sizes that the various models will need to incorporate to accomodate different 
      downsampling factors. 
    - downsampling_factors: The different downsampling_factors. 
    - lr: The learning rate. 
    - num_epochs - The amount of epochs to train for. 
    - batch_size - Size of the batches. 
    - loss - The loss metric used to update the weights. 

  Output: 
    - Trained a saved models with appropriate names. 
  
  """
  print('COMMENCE THE TRAINING OF THE MLPs')
  for idx, (in_size, downsampling_factor) in enumerate(zip(input_sizes, downsampling_factors)):
    train_set = SliceLoader(downsampling_factor)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True)   
    
    net = createMLP(in_size)

    net.apply(weight_init);
    updater = torch.optim.Adam(net.parameters(), lr=lr) 

    for epoch in range(num_epochs): 
      for X, y in train_loader:
        # Compute gradients and update parameters
        if device == 'cuda': 
          net.cuda()   
          X = X.cuda()
          y = y.cuda()
         
        y_hat = net(X[:,:,:].reshape(len(X),-1))
        l = loss(y_hat, y[:,:,:].reshape(len(y),-1))

        updater.zero_grad()
        l.mean().backward()
        updater.step()

      if epoch % 1 == 0: 
        print(f'Epoch: {epoch}, loss: {l.mean()}')
  
    torch.save(net, f'models/MLP{idx}') 





#############  CONVOLUTIONAL NEURAL NETWORKS #############   
def createCNN(in_size):
  """ createCNN

  A function which creates a convolutional neural network. 
  
  """
  net2 =  net2 = nn.Sequential(
          nn.Conv2d(1, 96, kernel_size=11, padding=1), nn.ReLU(), 
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
          nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
          nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Flatten(),
          nn.Linear(in_size, 4096), nn.ReLU(),
          nn.Dropout(p=0.3), 
          nn.Linear(4096, 192*160))
  return net2   

def trainCNNs(input_sizes, downsampling_factors, lr, num_epochs, batch_size, loss):
  """ trainCNNs
  
  A function which trains various CNNs based on the downsampling factors. 

  Args:
    - input_sizes: The input sizes that the various models will need to incorporate to accomodate different 
      downsampling factors. 
    - downsampling_factors: The different downsampling_factors. 
    - lr: The learning rate. 
    - num_epochs - The amount of epochs to train for. 
    - batch_size - Size of the batches. 
    - loss - The loss metric used to update the weights. 

  Output: 
    - Trained a saved models with appropriate names. 
  
  """
  print('COMMENCE THE TRAINING OF THE CNNs') 

  for idx, (in_size, downsampling_factor) in enumerate(zip(input_sizes, downsampling_factors)):
    train_set = SliceLoader(downsampling_factor = downsampling_factor)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,  
        batch_size=batch_size,  
        shuffle=True)   

    net2 = createCNN(in_size)

    # Hyperparameter settings  
    updater = torch.optim.Adam(net2.parameters(), lr=lr)

    for epoch in range(num_epochs): 
      for X, y in train_loader:
        # Compute gradients and update parameters
        if device == 'cuda': 
          net2.cuda() 
          X = X.cuda()
          y = y.cuda()
        
        
        y_hat = net2(X.unsqueeze(1))
         
        l = loss(y_hat, y[:,:,:].reshape(len(y),-1))

        updater.zero_grad()  
        l.mean().backward()
        updater.step()

      if epoch % 1 == 0: 
        print(f'Epoch: {epoch}, loss: {l.mean()}')

    torch.save(net2, f'models/CNN{idx}') 
    

        
       

#############  RESIDUALLY CONNECTED CONVOLUTIONAL NEURAL NETWORKS #############          
class CreateBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, id=None):
    super(CreateBlock, self).__init__()
    self.in_channels = in_channels 
    self.architecture = [[out_channels, 1, 0], [out_channels, 3,  1], [out_channels*4, 1, 0]]
    self.stride = stride 
    self.conv_layers = self.create_conv_layers() 
    self.ReLU = nn.ReLU() 
    self.id = id  
    

  def forward(self, x): 
    res = x.clone()   
    x = self.conv_layers(x) 

    if self.id is not None:
      res = self.id(res) 
    
    x = res + x 
    x = self.ReLU(x)   
    return x    

  def create_conv_layers(self):
    layers = list()
    in_channels = self.in_channels 
    
    for layer in self.architecture:
      out_channels = layer[0]    
      layers += [nn.Conv2d(in_channels, out_channels, kernel_size = layer[1], stride = self.stride, padding = layer[2]),
                nn.BatchNorm2d(out_channels), nn.ReLU()]
      
      in_channels = out_channels  
      
    return nn.Sequential(*layers) 

class ResNet(nn.Module):
  def __init__(self, CreateBlock, Blocks, Architecture, inSize):                    
    super(ResNet, self).__init__()
    self.CreateBlock = CreateBlock 
    self.init_layers = self.createLayers()
    self.in_channels = 64
    
    self.res_layers = self.createLayers(res_reps = Blocks, res_arch = Architecture)  

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
    self.fc = nn.Linear(inSize, 30720) 

  def forward(self, x):
    x = self.init_layers(x) 
    x = self.res_layers(x)  
    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    
    x = self.fc(x)
     
    return x 


  def createLayers(self, CreateBlock=CreateBlock, res_reps=None, res_arch=None): 
    if res_reps is not None:
      layers, id = list(), None

      for reps, arch in zip(res_reps, res_arch): 
        out_channels = arch[0]
        stride = arch[1] 

        if self.in_channels != out_channels * 4:
          id = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride = stride), 
                            nn.BatchNorm2d(out_channels*4))  
        
        layers.append(CreateBlock(self.in_channels, out_channels, stride=stride, id=id))
        self.in_channels = out_channels * 4 

        for i in range(reps - 1): 
          layers.append(CreateBlock(self.in_channels, out_channels))
      
    else:
      layers = [nn.Conv2d(1, 64, kernel_size = 7, stride =2, padding=3), nn.BatchNorm2d(64),
                nn.ReLU(), nn.MaxPool2d(kernel_size =3, stride=2, padding =1)]

    return nn.Sequential(*layers)     


def createRN(in_size):
  """ createRN

  A function which creates a convolutional neural network with skip connections (ResNet)
  
  """
  return ResNet(CreateBlock, [2, 3, 5, 2], [[128, 1], [256, 1], [512, 1]], in_size) 

def trainRNs(input_sizes, downsampling_factors, lr, num_epochs, batch_size, loss):
  """ trainRNs
  
  A function which trains various ResNets based on the downsampling factors. 

  Args:
    - input_sizes: The input sizes that the various models will need to incorporate to accomodate different 
      downsampling factors. 
    - downsampling_factors: The different downsampling_factors. 
    - lr: The learning rate. 
    - num_epochs - The amount of epochs to train for. 
    - batch_size - Size of the batches. 
    - loss - The loss metric used to update the weights. 

  Output: 
    - Trained a saved models with appropriate names. 
  
  """

  print('COMMENCE THE TRAINING OF THE RES NETs')
  
  for idx, (in_size, downsampling_factor) in enumerate(zip(input_sizes, downsampling_factors)):
    train_set = SliceLoader(downsampling_factor)
    
    train_loader = torch.utils.data.DataLoader( 
        train_set, 
        batch_size=batch_size, 
        shuffle=True)   
    
    
   
    model = createRN(in_size)
    model.apply(weight_initialisation);  
  
    updater = torch.optim.Adam(model.parameters(), lr=lr) 

    if device == 'cuda': 
      model.cuda()        
         
    for epoch in range(num_epochs):   
      iteration = 0 
    
      for low_res, high_res in train_loader:  
      
        if device == 'cuda': #shift
            low_res, high_res = low_res.cuda(), high_res.cuda()

        low_res = torch.unsqueeze(low_res, 1)  
        y_hat = model(low_res) 
        
        high_res = high_res.view(high_res.size(0), -1) 
        
        l = loss(y_hat, high_res)  
                                      
        updater.zero_grad()   
        torch.autograd.set_detect_anomaly(True)    
        l.backward()  
      
        updater.step()  
    
        iteration += 1      
      
        
    print('[Epoch %d, iter %05d] loss: %.9f' % (epoch, iteration, l.item()))
    
  
    torch.save(model, f'models/RN{idx}') 

############# GENERATIVE ADVERSARIAL NETWORKS ############# 
class GANBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, id=None):
    super(CreateBlock, self).__init__()
    self.in_channels = in_channels 
    self.out_channels = out_channels 
    self.stride = stride 
    self.conv_layers = self.create_conv_layer() 
    self.ReLU = nn.ReLU() 
    self.id = id  
    

  def forward(self, x): 
    res = x.clone()   
    x = self.conv_layers(x) 

    if self.id is not None:
      res = self.id(res) 
    
    x = torch.sum(x, dim=0)  
    x = res + x 
    x = self.ReLU(x)   
    return x    

  def create_conv_layer(self):
    layers = list()
    in_channels = self.in_channels 
    
    for i in range(2):  
      layers += [nn.Conv2d(self.in_channels, self.out_channels, kernel_size = 3, stride = 1),
                  nn.BatchNorm2d(self.out_channels), nn.PReLU()]
    

    return nn.Sequential(*layers) 

class GENERATOR(nn.Module):
  def __init__(self, GANBlock, Block, Architecture):                    
    super(ResNet, self).__init__()
    self.GANBlock = GANBlock
    self.init_layers = self.createLayers(init=True) 
    self.in_channels = 1 
    self.res_layers = self.createLayers(res_reps = Block, res_arch = Architecture)  
    self.conv_layer = nn.Sequential(nn.Conv2d(64, 64, 3, 1), BatchNorm2d(64)) 
    self.final_layers = self.createLayers()
    

  def forward(self, x):
    x = self.init_layers(x) 
    res = x.clone() 
    x = self.res_layers(x)
    x = self.conv_layer(x)
    x = torch.sum(x, dim=0)  
    
    x = res + x  
    x = self.final_layers(x) 
    return x 


  def createLayers(self, GANBlock=GANBlock, res_reps=None, out_channels=None, init=None): 
    layers = list()
    
    if res_reps is not None:
      layers = None

      if self.in_channels != out_channels * 4:
          id = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride = stride), 
                          nn.BatchNorm2d(out_channels*4))  
      
      layers.append(GANBlock(self.in_channels, out_channels, stride=1, id=id))
      
      in_channels = out_channels  

      for i in range(res_reps - 1): 
        layers.append(GANBlock(in_channels, out_channels))
       
    elif init is not None:
      layers.append([nn.Conv2d(1, 64, kernel_size = 9, stride =1), nn.BatchNorm2d(64),
                nn.PReLU(),])

    else:
      in_channels = 64 
      for i in range(2):
        layers.append(nn.Conv2d(in_channels, 256, kernel_size=3, stride=1), nn.PixelShuffle(2), 
                      nn.PReLU()) 
      
      layers.append(nn.Conv2d(256, 1, kernel_size=9, stride=1))
        

    return nn.Sequential(*layers)   


class DISCRIMINATOR(nn.Module):
  def __init__(self, in_channels, Architecture):
    self.in_channels = in_channels
    self.conv_layers = self.createLayers(Architecture)
    self.fc = nn.Sequential(nn.Linear(3000, 1024), 
                            nn.LeakyReLU(), 
                            nn.Linear(1024, 1), 
                            nn.Sigmoid())

  def forward(self, x):
    x = self.conv_layers(x) 
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)
    
    return x 


  def createLayers(self, Architecture):
    in_channels, layers = self.in_channels, list()

    for idx, layer in enumerate(Architecture):
      stride = 2 if idx % 2 == 0 else 1 
      out_channels = layer 
      
      layers += [nn.ConvTranspose2d(in_channels, out_channels, 4, stride),
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(True)] 
    
     
      in_channels = out_channels 
    
    layers.extend([nn.AdaptivePool(192, 160), nn.Sigmoid()]) 
    
    return nn.Sequential(*layers) 



def createGAN():
    """
    createGAN

    A function which creates a generative adversarial network with a particular input size. 

    Args:
        - in_size: The input size of the low resolution image which will be upsampled. 

    Output:
        - generator: The Generator network of the GAN
        - discriminator: The discriminator network of the GAN 
    
    """
    generator = GENERATOR(GANBlock, 5, 64) 
    discriminator = DISCRIMINATOR(1, createArchitecture())
    discriminator.apply(weights_init) 

    return generator, discriminator

def trainGAN(Discriminator, Generator, num_epochs, loss, train_loader, discrUpdater, genUpdater):
    img_list, g_loss, d_loss = [], [], [] 
    true, fake = 1, 0  

    fig, ax = plt.subplots(1, 3, figsize = (10, 10)) 
    ax = ax.flatten() 

    # os.mkdir('/content/model') 
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    if device == 'cuda': 
        Discriminator.cuda(), Generator.cuda()   
            
    for epoch in tqdm(range(num_epochs)):   
        iteration = 0 
    
    for low_res, high_res in train_loader:  
        if device == 'cuda':
            low_res, high_res = low_res.cuda(), high_res.cuda() 
        
        Discriminator.zero_grad()

        lenLab = len(high_res.view(-1))

        label = torch.full((lenLab,), true, dtype=torch.float, device=device)
        
        high_res = torch.unsqueeze(high_res, 1) 
        
        y_hat = Discriminator(high_res).view(-1) 
        
        realDl = loss(y_hat, label)  
    
        realDl.backward()
    
        avg = realDl.mean().item()  

        ############################
    
        #train with fake batch 
        low_res = torch.unsqueeze(low_res, 1)
        
        pred = Generator(low_res) 

        # pred.shape  
        
        label.fill_(fake) 
        
        y_hat = Generator(pred.detach()).view(-1) 

        fakeDl = loss(y_hat, label)

        fakeDl.backward() 
    
        avg1 = y_hat.mean().item() 

        #discriminator error as real + fake batch losses

        dLoss = realDl + fakeDl 

        discrUpdater.step() 

        ##############################

        Generator.zero_grad() 
        label.fill_(true) 

        y_hat = Discriminator(pred).view(-1)

        gLoss = loss(y_hat, label)

        gLoss.backward() 

        avg2 = y_hat.mean().item() 

        genUpdater.step() 


        # Output training stats
        if iteration % 50 == 0:
            print(iteration)
            # print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #       % (epoch, num_epochs, iteration, len(train_loader),
            #           dLoss.item(), gLoss.item(), avg, avg1, avg2))

            feature, label = next(iter(train_loader))
            fig, ax = plt.subplots(1, 3, figsize = (10, 10)) 
            ax = ax.flatten()   
            ax[0].imshow(feature[0], cmap= 'gray')
            ax[0].set_title('Low Res Image')
            ax[1].imshow(label[0], cmap = 'gray')  
            ax[1].set_title('High Res Label') 
            ax[2].imshow(Generator(torch.unsqueeze(torch.unsqueeze(feature[0], 0), 0).cuda()).cuda().cpu().detach().numpy().reshape(192, 160),  cmap='gray') 
            ax[2].set_title("Model's Prediction") 
            plt.pause(0.5) 
        plt.show()
            
            
        
        if epoch % 10 == 0:
            torch.save(Generator, '/content/model/generator') 
            torch.save(Discriminator, '/content/model/discriminator')
                
        g_loss.append(gLoss.item())
        d_loss.append(dLoss.item()) 

        if (iteration % 500 == 0) or ((epoch == num_epochs-1) and (iteration == len(train_loader)-1)):
            with torch.no_grad():
                flake = Generator(low_res).detach().cpu()
            img_list.append(vutils.make_grid(flake, padding=2, normalize=True))


        iteration += 1    