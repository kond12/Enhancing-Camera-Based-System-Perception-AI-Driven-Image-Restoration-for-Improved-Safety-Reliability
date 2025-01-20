### train the network
import os
from tkinter import Variable
import torchvision.utils as vutils
#import nntplib
import numpy as np
from plotting import plot_metrics_and_loss, save_metrics_and_losses_to_csv
#import pandas as pd
import torch
import torchvision.transforms as F
from matplotlib import pyplot as plt
from numpy.random import rand
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch.optim as optim
from mean_std_methods import *
from torch import nn
from torch import device
from torchvision.utils import save_image
from torchvision import transforms, utils
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Create the "output_images" directory if it doesn't exist
os.makedirs("output_images", exist_ok=True)


#from main import train_data
def calculate_metrics(original, denoised):
    # Convert tensors to NumPy arrays and adjust their ranges if necessary
    original = original.detach().cpu().numpy()
    denoised = denoised.detach().cpu().numpy()

   
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(original, denoised, data_range=1.0)

    ssim = structural_similarity(original, denoised, data_range=1.0,channel_axis=-1,win_size=3)


    # Calculate MSE
    mse = mean_squared_error(original.flatten(), denoised.flatten())

    # Calculate MAE
    mae = mean_absolute_error(original.flatten(), denoised.flatten())

    return psnr, ssim, mse, mae

def store_sample_image(device, model,  test_data_loader, epoch, psnr_list, ssim_list, mse_list, mae_list):
 if epoch % 1 == 0:
    c_images, f_images = next(iter(test_data_loader))
    c_images = c_images.to(device=device)
    f_images = f_images.to(device=device)
                # Output of reconstructed generated image
    with torch.no_grad():
        #output = model(f_images)
        output = model(f_images[:c_images.size(0)]) 
        output_normalized = (output - output.min()) / (output.max() - output.min())
        #output_normalized = (output - output.min()) / (output.max() - output.min())
        psnr, ssim, mse, mae = calculate_metrics(c_images, output_normalized)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mse_list.append(mse)
        mae_list.append(mae)
        

    # Store the output data to disk
    sample = torch.cat((c_images, f_images, output_normalized), -2)
    #sample1 = torch.cat((c_images, f_images, output_normalized), -1)
    vutils.save_image(sample, "output_images/%d_V.png" % epoch, nrow=6, normalize=False)
    #vutils.save_image(sample1, "output_images/%d_H.png" % epoch, nrow=6, normalize=False)
    print(f"Epoch {epoch}: PSNR={psnr:.2f}\nSSIM={ssim:.4f}\nMSE={mse:.4f}\nMAE={mae:.4f}")

            
   
def train(
    epochs,
    model,
    device,
    batch_size,
    train_loader,
    valid_loader,
    test_loader,
    loss_function,
    optimizer
):
    
    losses = []  # list containing average loss of each epoch
    train_losses = []
    valid_losses = []
    psnr_list = []
    ssim_list = []
    mse_list = []
    mae_list = []
    running_loss = 0
    l_train = len(train_loader) * batch_size
    l_valid = len(valid_loader)

    min_val_loss = np.inf
    test_batch = next(iter(test_loader))
    c_images_test, f_images_test = test_batch
    c_images_test = c_images_test.to(device)
    f_images_test = f_images_test.to(device)
    test_data_loader = DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False)
    # training loop
    model = model.to(device)
    loss_function= loss_function.to(device)
    for epoch in range(epochs):
        flag = True
        train_loss = 0.0
        model.train()
        #  loop
        for c_images, f_images in train_loader:
        #for i, (c_images, f_images) in enumerate(train_loader):
        #for i, (c_images, f_images) in enumerate(train_loader, total=int((l) / dataloader.batch_size)):
            c_images = c_images.to(device)
            f_images = f_images.to(device)  # shifting noisy images to cuda

            optimizer.zero_grad()
            output = model(f_images)
            
            loss = loss_function(output, c_images)  # loss is calculated between model output and og dataset images
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
    
        
        
        
        # validation loop
        model.eval()
        valid_loss = 0.0
        for c_images, f_images in valid_loader:
        #for i, (c_images, f_images) in enumerate(train_loader, total=int((l) / dataloader.batch_size)):
            c_images = c_images.to(device)
            f_images = f_images.to(device)  # shifting noisy images to cuda

            model.zero_grad()
            output = model(f_images)
            loss = loss_function(output, c_images)  # loss is calculated between model output and og dataset images

            valid_loss += loss.item()

           

        #losses.append([train_loss/ l_train,valid_loss / l_valid])
        train_losses.append(train_loss /l_train )
        valid_losses.append(valid_loss / l_valid)
        print('Epoch ',epoch,': Training loss:',train_loss/ l_train,'/t Validation loss: ',valid_loss / l_valid)
        '''psnr, ssim, mse, mae = calculate_metrics(c_images_test, output)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mse_list.append(mse)
        mae_list.append(mae)'''
            # Saving one image sample per epoch for visual feedback
        #savetestsamples(device, model,test_loader, epoch)
        if epoch % 1 == 0:
            store_sample_image(device, model,  test_data_loader, epoch, psnr_list, ssim_list, mse_list, mae_list)

        print(f"Epoch {epoch}:")
        print("Length of epochs range:", len(range(epochs)))
        print("Length of psnr_list:", len(psnr_list))
        print("Length of ssim_list:", len(ssim_list))
        print("Length of mse_list:", len(mse_list))
        print("Length of mae_list:", len(mae_list))

        #store_sample_image(device,model,test_loader,epoch)
        
        # checking the minimum validation loss and saving the model of the minimum loss
        if min_val_loss > valid_loss:
            print(f'Validation loss decreased ({min_val_loss:.6f}---->{valid_loss:.6f})/n Saving the Model withh the minimum loss')
            min_val_loss = valid_loss # updating the minimum loss
            # Saving the model
            torch.save(model.state_dict(),'model_'+str(epoch)+'.pth')
            #torch.save(model.state_dict(), f'model_{epoch}.pth')
            #torch.save(model.state_dict(),'model.pth')
       ## if epoch % 10 == 0:
         ##   savetestsamples(test_loader, epoch)
#plot_metrics_and_loss(epochs, train_losses, valid_losses, psnr_list, ssim_list, mse_list, mae_list)
    plot_metrics_and_loss(epochs, train_losses, valid_losses, psnr_list, ssim_list, mse_list, mae_list)
    csv_filename = "metrics_and_losses.csv"
    save_metrics_and_losses_to_csv(epochs, train_losses, valid_losses, psnr_list, ssim_list, mse_list, mae_list, csv_filename)
  
#4775018