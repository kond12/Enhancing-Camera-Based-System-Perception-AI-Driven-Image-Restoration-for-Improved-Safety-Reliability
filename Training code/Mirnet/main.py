import os
# from import data_loaders
# import pandas as pd
#from ploting_results import showtestsample
# from prepearing_data import prepare_data
import torch
from torchvision import transforms

# from torchvision.io import read_image, ImageReadMode
from mean_std_methods import *
from torch import nn
# from prepearing_data import *
from model1 import Autoencoders,Srcnn,MIRNet

from Training import train
from data import CustomImageDataset, data_loaders

print(torch.cuda.is_available())
# from data import train_data,test_data 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# test should be the same as train
current_path = (os.getcwd()).replace('\\', '/')  # './' also current directory
dataset_folder = '/MainDataset'
dataset_folder_path = current_path + dataset_folder

# change these two to reach different data
no_fault_folder = '/NoFault/'
#fault_folder = '/SpeckleNoise/'
fault_folder = '/Blur/'  # also Rain or SpeckleNoise
#fault_folder = '/Blur/'
#fault_folder = '/Rain/'
intensity_folder = '/m/'  # e as extreme also m as medium or l as low
# specify the csv file name
#csv_file = 'Rain_e_label.csv'
#csv_file = 'SpeckleNoise_l_label.csv'
#csv_file = 'Blur_l_label.csv'
csv_file = 'Blur_m_label.csv'
Batch_size = 12
EPOCHS = 250

if __name__ == '__main__':
    mean_f, std_f = get_mean(dataset_folder_path + fault_folder + intensity_folder), get_std(
        dataset_folder_path + fault_folder + intensity_folder)
    mean_c, std_c = get_mean(dataset_folder_path + no_fault_folder), get_std(dataset_folder_path + no_fault_folder)

    transform_f = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_f, std=std_f), ])

    # Normalize the test set same as training set without augmentation
    transform_c = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_c, std=std_c), ])

    # HERE LOADING DATA IS AFFECTING TO OUR NEW DATA + PATH ALSO NEED TO BE CHECKED IN ADDITION TO FILE NAMES
    # data = CustomImageDataset(dataset_folder_path + fault_folder + csv_file, transform=transform_f,target_transform=transform_c)

    train_loader, valid_loader, test_loader = data_loaders(
        CustomImageDataset(dataset_folder_path + fault_folder + csv_file, transform=transform_f,
                           target_transform=transform_c))
    model = MIRNet()

    # print(model)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(epochs=EPOCHS, model=model, device=device, batch_size=Batch_size,
          train_loader=train_loader, valid_loader=valid_loader,test_loader=test_loader, loss_function=loss_function, optimizer=optimizer)

   # showtestsample(test_loader)

# save_plots(train_loss, val_loss) # accuracy
