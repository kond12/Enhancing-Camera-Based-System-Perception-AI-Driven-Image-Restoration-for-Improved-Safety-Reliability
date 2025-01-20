import os
import numpy as np
import pandas as pd
from torchvision import transforms, io
from matplotlib import pyplot as plt
from numpy.random import rand
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torch.optim as optim
from mean_std_methods import *

def data_loaders(data, batch_size=12, test_size = 0.2, valid_size = 0.1, if_validation = True, num_workers=0):

    if if_validation:
        indices = list(range(len(data)))
        split1 = int(np.floor(test_size * len(data)))
        split2 = split1 +  int(np.floor(valid_size * len(data)))
        test_idx = indices[:split1]
        test_sampler = SubsetRandomSampler(test_idx)
        test_loader = DataLoader(data, num_workers=num_workers, batch_size=batch_size, sampler=test_sampler)
        np.random.shuffle(indices)

        train_idx = indices[split2:]  # mean from split2 value for example 0.3 of our data length means 30 out of 100, so from 30 till the end, which mean 70 samples
        valid_idx = indices[split1:split2]  # mean from split1 value till split1 value for example 0.1 till 0.3 of our data length means from 10 till 30 out of 100, so  20 samples
          # mean from begining till split1 value for example 0.1 of our data length means 10 out of 100, so from 0 till the 10, which mean 10 samples

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(valid_idx)
        

        train_loader = DataLoader(data, num_workers=num_workers, batch_size=batch_size,  sampler=train_sampler)
        valid_loader = DataLoader(data, num_workers=num_workers, batch_size=batch_size, sampler=validation_sampler)
        
        return train_loader, valid_loader, test_loader
    else:
        indices = list(range(len(data)))
        split1 = int(np.floor(test_size * len(data)))
        test_idx = indices[:split1]
        test_sampler = SubsetRandomSampler(test_idx)
        test_loader = DataLoader(data, num_workers=num_workers, batch_size=1, sampler=test_sampler)
        np.random.shuffle(indices)

        train_idx = indices[split1:]
        

        train_sampler = SubsetRandomSampler(train_idx)
        

        train_loader = DataLoader(data, num_workers=num_workers, batch_size=batch_size,  sampler=train_sampler)
        return train_loader, test_loader

# clear is our target/label and the input is noised image
# corrected or predicted result is the output from the network after processing the noisy input image to make it more similar or identical to the target



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        self.show = False

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # picture_name = self.img_labels.iloc[idx, 0]
        c_path = self.img_labels.iloc[idx, 3]
        f_path = self.img_labels.iloc[idx, 4]

        c_image = io.read_image(c_path)
        f_image = io.read_image(f_path)

        if self.transform:
            transform = transforms.ToPILImage()
            # convert the tensor to PIL image using above transform
            f_image = transform(f_image)
            f_image = f_image.convert('RGB')
            f_image = self.transform(f_image)
        if self.target_transform:
            transform = transforms.ToPILImage()
            # convert the tensor to PIL image using above transform
            c_image = transform(c_image)
            c_image = c_image.convert('RGB')
            c_image = self.transform(c_image)
        return c_image, f_image

