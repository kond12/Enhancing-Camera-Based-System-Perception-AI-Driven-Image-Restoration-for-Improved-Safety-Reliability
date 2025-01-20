import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv
# Set environment variable to avoid 'KMP_DUPLICATE_LIB_OK' issue on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the transformation to convert images to tensors and vice versa
transform = transforms.Compose([transforms.ToTensor(), ])
image_transform = transforms.Compose([transforms.ToPILImage(), ])

class Autoencoders(nn.Module):
    def __init__(self):
        super(Autoencoders, self).__init__()
        # encoding layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=2)  # 28x28 --> 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)  # 14x14 --> 7x7
        self.conv3 = nn.Conv2d(64, 128, 5)  # 7x7 --> 3x3

        # decoding layers
        self.tconv1 = nn.ConvTranspose2d(128, 64, 5)  # 3x3 --> 7x7
        self.tconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 7x7 --> 14x14
        self.tconv3 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)  # 14x14 --> 28x28

        self.relu = torch.nn.ReLU()
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.relu(self.tconv1(x))
        x = self.relu(self.tconv2(x))
        x = self.activation(self.tconv3(x))  # final layer is applied sigmoid activation

        return x

# List of model checkpoint file paths
model_paths = [
    'C:/ablur/model_0.pth',
    'C:/ablur/model_1.pth',
    'C:/ablur/model_2.pth',
    'C:/ablur/model_3.pth',
    'C:/ablur/model_4.pth',
    'C:/ablur/model_5.pth',
    'C:/ablur/model_6.pth',
    'C:/ablur/model_17.pth',
    'C:/ablur/model_23.pth',
    'C:/ablur/model_38.pth',
    'C:/ablur/model_50.pth',
    'C:/ablur/model_71.pth',
    'C:/ablur/model_73.pth',
    'C:/ablur/model_110.pth',
    'C:/ablur/model_144.pth',
    'C:/ablur/model_152.pth',
    'C:/ablur/model_195.pth',
    
    
    # Add more model paths as needed
]

# Image file path  16_SpeckleNoise_e
image_path = 'C:/16_Blur_e.jpg'

# Output directory
output_dir = 'test_output_imagesabe'
os.makedirs(output_dir, exist_ok=True)

# Load the image
image_path1 = 'C:/16_NoFault.jpg'
metrics_data = []

# Iterate through models and test on the same image
for model_path in model_paths:
    # Instantiate the model
    model = Autoencoders()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the noisy image
    noisy_image = Image.open(image_path)
    noisy_tensor = transform(noisy_image)
    clear_image = Image.open(image_path1)
    clear_tensor = transform(clear_image)

    # Perform denoising using the model
    with torch.no_grad():
        output = model(noisy_tensor.unsqueeze(0))  # Add batch dimension
    out_norm = (output - output.min()) / (output.max() - output.min())
    denoised_image = image_transform(out_norm.squeeze())  # Remove batch dimension and convert to PIL Image

    # Create a composite image
    composite = Image.new('RGB', (clear_image.width * 3, clear_image.height))
    composite.paste(clear_image, (0, 0))  # Clear image on the left
    composite.paste(noisy_image, (clear_image.width, 0))  # Noisy image in the middle
    composite.paste(denoised_image, (clear_image.width * 2, 0))  # Denoised image on the right


    # Save the composite image
    output_filename = f'composite_autoencoder_{os.path.basename(model_path)}_{os.path.basename(image_path)}'
    composite.save(os.path.join(output_dir, output_filename))

    # Convert PIL Image objects to NumPy arrays
    noisy_np = np.array(noisy_image)
    denoised_np = np.array(denoised_image)
    clear_np = np.array(clear_image)

    # Calculate metrics (you can add your metric calculation here if needed)
    # Calculate metrics
    psnr = peak_signal_noise_ratio(clear_np, denoised_np)
    ssim_value = ssim(clear_np, denoised_np, win_size=11, channel_axis=-1)  # Adjust win_size as needed
    mse = mean_squared_error(clear_np.flatten(), denoised_np.flatten())
    mae = mean_absolute_error(clear_np.flatten(), denoised_np.flatten())
    
            # Print or save the metrics as needed
    print(f' Model Path: {model_path}, Image: {image_path}')
    print(f'PSNR: {psnr:.2f}, SSIM: {ssim_value:.2f}, MSE: {mse:.2f}, MAE: {mae:.2f}')
    print('---')

            # Save metrics to the metrics_data list, converting None or np.nan values to placeholders
    metrics_data.append([model_path, image_path, psnr, ssim_value if not np.isnan(ssim_value) else 'N/A', mse if not np.isnan(mse) else 'N/A', mae if not np.isnan(mae) else 'N/A'])

# Save metrics data to CSV file
csv_filename = 'metrics.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header
    csv_writer.writerow(['Model Path', 'Image', 'PSNR', 'SSIM', 'MSE', 'MAE'])
    # Write metrics data
    csv_writer.writerows(metrics_data)

print(f'Metrics data saved to {csv_filename}')


    # Print or save the metrics as needed
print(f'Model: Autoencoders, Model Path: {model_path}, Image: {image_path}')
print('---')

print(f'Composite images saved to {output_dir}')


import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# Set environment variable to avoid 'KMP_DUPLICATE_LIB_OK' issue on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the transformation to convert images to tensors and vice versa
transform = transforms.Compose([transforms.ToTensor(), ])
image_transform = transforms.Compose([transforms.ToPILImage(), ])

class Autoencoders(nn.Module):
    def __init__(self):
        super(Autoencoders, self).__init__()
        # encoding layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=2)  # 28x28 --> 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)  # 14x14 --> 7x7
        self.conv3 = nn.Conv2d(64, 128, 5)  # 7x7 --> 3x3

        # decoding layers
        self.tconv1 = nn.ConvTranspose2d(128, 64, 5)  # 3x3 --> 7x7
        self.tconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 7x7 --> 14x14
        self.tconv3 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)  # 14x14 --> 28x28

        self.relu = torch.nn.ReLU()
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.relu(self.tconv1(x))
        x = self.relu(self.tconv2(x))
        x = self.activation(self.tconv3(x))  # final layer is applied sigmoid activation

        return x

# List of model checkpoint file paths
model_paths = [
    'C:/anoise/model_0.pth',
    'C:/anoise/model_1.pth',
    'C:/anoise/model_2.pth',
    'C:/anoise/model_3.pth',
    'C:/anoise/model_4.pth',
    'C:/anoise/model_5.pth',
    'C:/anoise/model_6.pth',
    'C:/anoise/model_8.pth',
    'C:/anoise/model_9.pth',
    'C:/anoise/model_10.pth',
    'C:/anoise/model_12.pth',
    'C:/anoise/model_13.pth',
    'C:/anoise/model_15.pth',
    'C:/anoise/model_20.pth',
    'C:/anoise/model_23.pth',
    'C:/anoise/model_27.pth',
    'C:/anoise/model_54.pth',
    'C:/anoise/model_61.pth',
    'C:/anoise/model_77.pth',
    'C:/anoise/model_78.pth',
    'C:/anoise/model_86.pth',
    'C:/anoise/model_92.pth',
    'C:/anoise/model_105.pth',
    'C:/anoise/model_114.pth',
    'C:/anoise/model_201.pth'
    # Add more model paths as needed
]

# Image file path
image_path = 'C:/16_SpeckleNoise_l.jpg'

# Output directory
output_dir = 'test_output_images-anl'
os.makedirs(output_dir, exist_ok=True)

# Load the image
clear_image = Image.open('C:/16_NoFault.jpg')

# Iterate through models and test on the same image
for model_path in model_paths:
    # Instantiate the model
    model = Autoencoders()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the noisy image
    noisy_image = Image.open(image_path)
    noisy_tensor = transform(noisy_image)

    # Perform denoising using the model
    with torch.no_grad():
        output = model(noisy_tensor.unsqueeze(0))  # Add batch dimension
    out_norm = (output - output.min()) / (output.max() - output.min())
    denoised_image = image_transform(out_norm.squeeze())  # Remove batch dimension and convert to PIL Image

    # Create a composite image
    composite = Image.new('RGB', (clear_image.width * 3, clear_image.height))
    composite.paste(clear_image, (0, 0))  # Clear image on the left
    composite.paste(noisy_image, (clear_image.width, 0))  # Noisy image in the middle
    composite.paste(denoised_image, (clear_image.width * 2, 0))  # Denoised image on the right


    # Save the composite image
    output_filename = f'composite_autoencoder_{os.path.basename(model_path)}_{os.path.basename(image_path)}'
    composite.save(os.path.join(output_dir, output_filename))

    # Convert PIL Image objects to NumPy arrays
    noisy_np = np.array(noisy_image)
    denoised_np = np.array(denoised_image)

    # Calculate metrics (you can add your metric calculation here if needed)

    # Print or save the metrics as needed
    print(f'Model: Autoencoders, Model Path: {model_path}, Image: {image_path}')
    print('---')

print(f'Composite images saved to {output_dir}')


import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# Set environment variable to avoid 'KMP_DUPLICATE_LIB_OK' issue on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the transformation to convert images to tensors and vice versa
transform = transforms.Compose([transforms.ToTensor(), ])
image_transform = transforms.Compose([transforms.ToPILImage(), ])

class Autoencoders(nn.Module):
    def __init__(self):
        super(Autoencoders, self).__init__()
        # encoding layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=2)  # 28x28 --> 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)  # 14x14 --> 7x7
        self.conv3 = nn.Conv2d(64, 128, 5)  # 7x7 --> 3x3

        # decoding layers
        self.tconv1 = nn.ConvTranspose2d(128, 64, 5)  # 3x3 --> 7x7
        self.tconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 7x7 --> 14x14
        self.tconv3 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)  # 14x14 --> 28x28

        self.relu = torch.nn.ReLU()
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.relu(self.tconv1(x))
        x = self.relu(self.tconv2(x))
        x = self.activation(self.tconv3(x))  # final layer is applied sigmoid activation

        return x

# List of model checkpoint file paths
model_paths = [
    'C:/arain/model_0.pth',
    'C:/arain/model_1.pth',
    'C:/arain/model_2.pth',
    'C:/arain/model_3.pth',
    'C:/arain/model_4.pth',
    'C:/arain/model_5.pth',
    'C:/arain/model_6.pth',
    'C:/arain/model_9.pth',
    'C:/arain/model_11.pth',
    'C:/arain/model_15.pth',
    'C:/arain/model_18.pth',
    'C:/arain/model_20.pth',
    'C:/arain/model_22.pth',
    'C:/arain/model_23.pth',
    'C:/arain/model_24.pth',
    'C:/arain/model_28.pth',
    'C:/arain/model_29.pth',
    'C:/arain/model_34.pth',
    'C:/arain/model_50.pth',
    'C:/arain/model_73.pth',
    'C:/arain/model_74.pth',
    'C:/arain/model_86.pth',
    'C:/arain/model_103.pth',
    'C:/arain/model_117.pth',
    'C:/arain/model_121.pth',
    'C:/arain/model_166.pth',
    'C:/arain/model_212.pth',
    'C:/arain/model_234.pth',
    'C:/arain/model_243.pth'
    # Add more model paths as needed
]

# Image file path
image_path = 'C:/16_Rain_l.jpg'

# Output directory
output_dir = 'test_output_imagesarl'
os.makedirs(output_dir, exist_ok=True)

# Load the image
clear_image = Image.open('C:/16_NoFault.jpg')

# Iterate through models and test on the same image
for model_path in model_paths:
    # Instantiate the model
    model = Autoencoders()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the noisy image
    noisy_image = Image.open(image_path)
    noisy_tensor = transform(noisy_image)

    # Perform denoising using the model
    with torch.no_grad():
        output = model(noisy_tensor.unsqueeze(0))  # Add batch dimension
    out_norm = (output - output.min()) / (output.max() - output.min())
    denoised_image = image_transform(out_norm.squeeze())  # Remove batch dimension and convert to PIL Image

    # Create a composite image
    composite = Image.new('RGB', (clear_image.width * 3, clear_image.height))
    composite.paste(clear_image, (0, 0))  # Clear image on the left
    composite.paste(noisy_image, (clear_image.width, 0))  # Noisy image in the middle
    composite.paste(denoised_image, (clear_image.width * 2, 0))  # Denoised image on the right


    # Save the composite image
    output_filename = f'composite_autoencoder_{os.path.basename(model_path)}_{os.path.basename(image_path)}'
    composite.save(os.path.join(output_dir, output_filename))

    # Convert PIL Image objects to NumPy arrays
    noisy_np = np.array(noisy_image)
    denoised_np = np.array(denoised_image)

    # Calculate metrics (you can add your metric calculation here if needed)

    # Print or save the metrics as needed
    print(f'Model: Autoencoders, Model Path: {model_path}, Image: {image_path}')
    print('---')

print(f'Composite images saved to {output_dir}')


import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# Set environment variable to avoid 'KMP_DUPLICATE_LIB_OK' issue on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the transformation to convert images to tensors and vice versa
transform = transforms.Compose([transforms.ToTensor(), ])
image_transform = transforms.Compose([transforms.ToPILImage(), ])

class MIRNet(nn.Module):
    def __init__(self):
        super(MIRNet, self).__init__()

        # Feature extraction network
        self.fe_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Information exchange network
        self.ien_net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # Selective kernel feature fusion network
        self.skff_net = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # Output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fe_net(x)
        x = self.ien_net(x)
        x = self.skff_net(x)
        x = self.output_layer(x)
        return x

# List of model checkpoint file paths
model_paths = [
    'C:/mblur/model_0.pth',
    'C:/mblur/model_1.pth',
    'C:/mblur/model_3.pth',
    'C:/mblur/model_5.pth',
    'C:/mblur/model_8.pth',
    'C:/mblur/model_9.pth',
    'C:/mblur/model_10.pth',
    'C:/mblur/model_15.pth',
    'C:/mblur/model_20.pth',
    'C:/mblur/model_29.pth',
    'C:/mblur/model_40.pth',
    'C:/mblur/model_47.pth',
    'C:/mblur/model_57.pth',
    'C:/mblur/model_60.pth',
    'C:/mblur/model_71.pth',
    'C:/mblur/model_84.pth',
    'C:/mblur/model_87.pth',
    'C:/mblur/model_102.pth',
    'C:/mblur/model_121.pth',
    'C:/mblur/model_137.pth'
    # Add more model paths as needed
]

# Image file path
image_path = 'C:/16_Blur_l.jpg'

# Output directory
output_dir = 'test_output_imagesmbl'
os.makedirs(output_dir, exist_ok=True)

# Load the image
clear_image = Image.open('C:/16_NoFault.jpg')

# Iterate through models and test on the same image
for model_path in model_paths:
    # Instantiate the model
    model = MIRNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the noisy image
    noisy_image = Image.open(image_path)
    noisy_tensor = transform(noisy_image)

    # Perform denoising using the model
    with torch.no_grad():
        output = model(noisy_tensor.unsqueeze(0))  # Add batch dimension
    out_norm = (output - output.min()) / (output.max() - output.min())
    denoised_image = image_transform(out_norm.squeeze())  # Remove batch dimension and convert to PIL Image

    # Create a composite image
    composite = Image.new('RGB', (clear_image.width * 3, clear_image.height))
    composite.paste(clear_image, (0, 0))  # Clear image on the left
    composite.paste(noisy_image, (clear_image.width, 0))  # Noisy image in the middle
    composite.paste(denoised_image, (clear_image.width * 2, 0))  # Denoised image on the right


    # Save the composite image
    output_filename = f'composite_autoencoder_{os.path.basename(model_path)}_{os.path.basename(image_path)}'
    composite.save(os.path.join(output_dir, output_filename))

    # Convert PIL Image objects to NumPy arrays
    noisy_np = np.array(noisy_image)
    denoised_np = np.array(denoised_image)

    # Calculate metrics (you can add your metric calculation here if needed)

    # Print or save the metrics as needed
    print(f'Model: Autoencoders, Model Path: {model_path}, Image: {image_path}')
    print('---')

print(f'Composite images saved to {output_dir}')


import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# Set environment variable to avoid 'KMP_DUPLICATE_LIB_OK' issue on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the transformation to convert images to tensors and vice versa
transform = transforms.Compose([transforms.ToTensor(), ])
image_transform = transforms.Compose([transforms.ToPILImage(), ])

class MIRNet(nn.Module):
    def __init__(self):
        super(MIRNet, self).__init__()

        # Feature extraction network
        self.fe_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Information exchange network
        self.ien_net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # Selective kernel feature fusion network
        self.skff_net = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # Output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fe_net(x)
        x = self.ien_net(x)
        x = self.skff_net(x)
        x = self.output_layer(x)
        return x


# List of model checkpoint file paths
model_paths = [
    'C:/mnoise/model_0.pth',
    'C:/mnoise/model_1.pth',
    'C:/mnoise/model_2.pth',
    'C:/mnoise/model_3.pth',
    'C:/mnoise/model_6.pth',
    'C:/mnoise/model_9.pth',
    'C:/mnoise/model_13.pth',
    'C:/mnoise/model_15.pth',
    'C:/mnoise/model_21.pth',
    'C:/mnoise/model_27.pth',
    'C:/mnoise/model_34.pth',
    'C:/mnoise/model_45.pth',
    'C:/mnoise/model_52.pth',
    'C:/mnoise/model_64.pth',
    'C:/mnoise/model_74.pth',
    'C:/mnoise/model_113.pth',
    'C:/mnoise/model_131.pth',
    'C:/mnoise/model_137.pth'
    
    # Add more model paths as needed
]

# Image file path
image_path = 'C:/16_SpeckleNoise_l.jpg'

# Output directory
output_dir = 'test_output_images-mnl'
os.makedirs(output_dir, exist_ok=True)

# Load the image
clear_image = Image.open('C:/16_NoFault.jpg')

# Iterate through models and test on the same image
for model_path in model_paths:
    # Instantiate the model
    model = MIRNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the noisy image
    noisy_image = Image.open(image_path)
    noisy_tensor = transform(noisy_image)

    # Perform denoising using the model
    with torch.no_grad():
        output = model(noisy_tensor.unsqueeze(0))  # Add batch dimension
    out_norm = (output - output.min()) / (output.max() - output.min())
    denoised_image = image_transform(out_norm.squeeze())  # Remove batch dimension and convert to PIL Image

    # Create a composite image
    composite = Image.new('RGB', (clear_image.width * 3, clear_image.height))
    composite.paste(clear_image, (0, 0))  # Clear image on the left
    composite.paste(noisy_image, (clear_image.width, 0))  # Noisy image in the middle
    composite.paste(denoised_image, (clear_image.width * 2, 0))  # Denoised image on the right


    # Save the composite image
    output_filename = f'composite_autoencoder_{os.path.basename(model_path)}_{os.path.basename(image_path)}'
    composite.save(os.path.join(output_dir, output_filename))

    # Convert PIL Image objects to NumPy arrays
    noisy_np = np.array(noisy_image)
    denoised_np = np.array(denoised_image)

    # Calculate metrics (you can add your metric calculation here if needed)

    # Print or save the metrics as needed
    print(f'Model: Autoencoders, Model Path: {model_path}, Image: {image_path}')
    print('---')

print(f'Composite images saved to {output_dir}')

import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# Set environment variable to avoid 'KMP_DUPLICATE_LIB_OK' issue on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the transformation to convert images to tensors and vice versa
transform = transforms.Compose([transforms.ToTensor(), ])
image_transform = transforms.Compose([transforms.ToPILImage(), ])

class MIRNet(nn.Module):
    def __init__(self):
        super(MIRNet, self).__init__()

        # Feature extraction network
        self.fe_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Information exchange network
        self.ien_net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # Selective kernel feature fusion network
        self.skff_net = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # Output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fe_net(x)
        x = self.ien_net(x)
        x = self.skff_net(x)
        x = self.output_layer(x)
        return x


# List of model checkpoint file paths
model_paths = [
    'C:/mrain/model_0.pth',
    'C:/mrain/model_1.pth',
    'C:/mrain/model_2.pth',
    'C:/mrain/model_3.pth',
    'C:/mrain/model_5.pth',
    'C:/mrain/model_9.pth',
    'C:/mrain/model_11.pth',
    'C:/mrain/model_13.pth',
    'C:/mrain/model_17.pth',
    'C:/mrain/model_22.pth',
    'C:/mrain/model_36.pth',
    'C:/mrain/model_45.pth',
    'C:/mrain/model_58.pth',
    'C:/mrain/model_68.pth',
    'C:/mrain/model_74.pth',
    'C:/mrain/model_85.pth',
    'C:/mrain/model_94.pth',
    'C:/mrain/model_111.pth',
    'C:/mrain/model_121.pth',
    'C:/mrain/model_123.pth',
    'C:/mrain/model_137.pth'
    # Add more model paths as needed
]

# Image file path
image_path = 'C:/16_Rain_l.jpg'

# Output directory
output_dir = 'test_output_images-mrl'
os.makedirs(output_dir, exist_ok=True)

# Load the image
clear_image = Image.open('C:/16_NoFault.jpg')

# Iterate through models and test on the same image
for model_path in model_paths:
    # Instantiate the model
    model = MIRNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the noisy image
    noisy_image = Image.open(image_path)
    noisy_tensor = transform(noisy_image)

    # Perform denoising using the model
    with torch.no_grad():
        output = model(noisy_tensor.unsqueeze(0))  # Add batch dimension
    out_norm = (output - output.min()) / (output.max() - output.min())
    denoised_image = image_transform(out_norm.squeeze())  # Remove batch dimension and convert to PIL Image

    # Create a composite image
    composite = Image.new('RGB', (clear_image.width * 3, clear_image.height))
    composite.paste(clear_image, (0, 0))  # Clear image on the left
    composite.paste(noisy_image, (clear_image.width, 0))  # Noisy image in the middle
    composite.paste(denoised_image, (clear_image.width * 2, 0))  # Denoised image on the right


    # Save the composite image
    output_filename = f'composite_autoencoder_{os.path.basename(model_path)}_{os.path.basename(image_path)}'
    composite.save(os.path.join(output_dir, output_filename))

    # Convert PIL Image objects to NumPy arrays
    noisy_np = np.array(noisy_image)
    denoised_np = np.array(denoised_image)

    # Calculate metrics (you can add your metric calculation here if needed)

    # Print or save the metrics as needed
    print(f'Model: Autoencoders, Model Path: {model_path}, Image: {image_path}')
    print('---')

print(f'Composite images saved to {output_dir}')


import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# Set environment variable to avoid 'KMP_DUPLICATE_LIB_OK' issue on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the transformation to convert images to tensors and vice versa
transform = transforms.Compose([transforms.ToTensor(), ])
image_transform = transforms.Compose([transforms.ToPILImage(), ])


class Srcnn(nn.Module):
    def __init__(self):
        super(Srcnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# List of model checkpoint file paths
model_paths = [
    'C:/sblur/model_0.pth',
    'C:/sblur/model_1.pth',
    'C:/sblur/model_3.pth',
    'C:/sblur/model_4.pth',
    'C:/sblur/model_6.pth',
    'C:/sblur/model_7.pth',
    'C:/sblur/model_8.pth',
    'C:/sblur/model_10.pth',
    'C:/sblur/model_14.pth',
    'C:/sblur/model_19.pth',
    'C:/sblur/model_21.pth',
    'C:/sblur/model_23.pth',
    'C:/sblur/model_28.pth',
    'C:/sblur/model_46.pth',
    'C:/sblur/model_65.pth',
    'C:/sblur/model_73.pth',
    'C:/sblur/model_86.pth',
    'C:/sblur/model_92.pth',
    'C:/sblur/model_96.pth',
    'C:/sblur/model_116.pth',
    'C:/sblur/model_132.pth',
    'C:/sblur/model_146.pth',
    'C:/sblur/model_187.pth',
    'C:/sblur/model_209.pth'
    # Add more model paths as needed
]

# Image file path
image_path = 'C:/16_Blur_l.jpg'

# Output directory
output_dir = 'test_output_images-sbl'
os.makedirs(output_dir, exist_ok=True)

# Load the image
clear_image = Image.open('C:/16_NoFault.jpg')

# Iterate through models and test on the same image
for model_path in model_paths:
    # Instantiate the model
    model = Srcnn()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the noisy image
    noisy_image = Image.open(image_path)
    noisy_tensor = transform(noisy_image)

    # Perform denoising using the model
    with torch.no_grad():
        output = model(noisy_tensor.unsqueeze(0))  # Add batch dimension
    out_norm = (output - output.min()) / (output.max() - output.min())
    denoised_image = image_transform(out_norm.squeeze())  # Remove batch dimension and convert to PIL Image

    # Create a composite image
    composite = Image.new('RGB', (clear_image.width * 3, clear_image.height))
    composite.paste(clear_image, (0, 0))  # Clear image on the left
    composite.paste(noisy_image, (clear_image.width, 0))  # Noisy image in the middle
    composite.paste(denoised_image, (clear_image.width * 2, 0))  # Denoised image on the right


    # Save the composite image
    output_filename = f'composite_autoencoder_{os.path.basename(model_path)}_{os.path.basename(image_path)}'
    composite.save(os.path.join(output_dir, output_filename))

    # Convert PIL Image objects to NumPy arrays
    noisy_np = np.array(noisy_image)
    denoised_np = np.array(denoised_image)

    # Calculate metrics (you can add your metric calculation here if needed)

    # Print or save the metrics as needed
    print(f'Model: Autoencoders, Model Path: {model_path}, Image: {image_path}')
    print('---')

print(f'Composite images saved to {output_dir}')


import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# Set environment variable to avoid 'KMP_DUPLICATE_LIB_OK' issue on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the transformation to convert images to tensors and vice versa
transform = transforms.Compose([transforms.ToTensor(), ])
image_transform = transforms.Compose([transforms.ToPILImage(), ])

class Srcnn(nn.Module):
    def __init__(self):
        super(Srcnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
# List of model checkpoint file paths
model_paths = [
    'C:/snoise/model_0.pth',
    'C:/snoise/model_1.pth',
    'C:/snoise/model_2.pth',
    'C:/snoise/model_3.pth',
    'C:/snoise/model_4.pth',
    'C:/snoise/model_6.pth',
    'C:/snoise/model_7.pth',
    'C:/snoise/model_17.pth',
    'C:/snoise/model_20.pth',
    'C:/snoise/model_28.pth',
    'C:/snoise/model_34.pth',
    'C:/snoise/model_39.pth',
    'C:/snoise/model_45.pth',
    'C:/snoise/model_55.pth',
    'C:/snoise/model_56.pth',
    'C:/snoise/model_68.pth',
    'C:/snoise/model_76.pth',
    'C:/snoise/model_93.pth',
    'C:/snoise/model_148.pth',

    'C:/snoise/model_176.pth',
    'C:/snoise/model_222.pth'
    # Add more model paths as needed
]

# Image file path
image_path = 'C:/16_SpeckleNoise_l.jpg'

# Output directory
output_dir = 'test_output_images-snl'
os.makedirs(output_dir, exist_ok=True)

# Load the image
clear_image = Image.open('C:/16_NoFault.jpg')

# Iterate through models and test on the same image
for model_path in model_paths:
    # Instantiate the model
    model = Srcnn()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the noisy image
    noisy_image = Image.open(image_path)
    noisy_tensor = transform(noisy_image)

    # Perform denoising using the model
    with torch.no_grad():
        output = model(noisy_tensor.unsqueeze(0))  # Add batch dimension
    out_norm = (output - output.min()) / (output.max() - output.min())
    denoised_image = image_transform(out_norm.squeeze())  # Remove batch dimension and convert to PIL Image

    # Create a composite image
    composite = Image.new('RGB', (clear_image.width * 3, clear_image.height))
    composite.paste(clear_image, (0, 0))  # Clear image on the left
    composite.paste(noisy_image, (clear_image.width, 0))  # Noisy image in the middle
    composite.paste(denoised_image, (clear_image.width * 2, 0))  # Denoised image on the right


    # Save the composite image
    output_filename = f'composite_autoencoder_{os.path.basename(model_path)}_{os.path.basename(image_path)}'
    composite.save(os.path.join(output_dir, output_filename))

    # Convert PIL Image objects to NumPy arrays
    noisy_np = np.array(noisy_image)
    denoised_np = np.array(denoised_image)

    # Calculate metrics (you can add your metric calculation here if needed)

    # Print or save the metrics as needed
    print(f'Model: Autoencoders, Model Path: {model_path}, Image: {image_path}')
    print('---')

print(f'Composite images saved to {output_dir}')

import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# Set environment variable to avoid 'KMP_DUPLICATE_LIB_OK' issue on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the transformation to convert images to tensors and vice versa
transform = transforms.Compose([transforms.ToTensor(), ])
image_transform = transforms.Compose([transforms.ToPILImage(), ])

class Srcnn(nn.Module):
    def __init__(self):
        super(Srcnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
# List of model checkpoint file paths
model_paths = [
    'C:/srain/model_0.pth',
    'C:/srain/model_1.pth',
    'C:/srain/model_2.pth',
    'C:/srain/model_3.pth',
    'C:/srain/model_4.pth',
    'C:/srain/model_5.pth',
    'C:/srain/model_7.pth',
    'C:/srain/model_14.pth',
    'C:/srain/model_17.pth',
    'C:/srain/model_19.pth',
    'C:/srain/model_32.pth',
    'C:/srain/model_40.pth',
    'C:/srain/model_54.pth',
    'C:/srain/model_68.pth',
    'C:/srain/model_87.pth',
    'C:/srain/model_116.pth',
    'C:/srain/model_127.pth',
    'C:/srain/model_132.pth',
    'C:/srain/model_142.pth',
    'C:/srain/model_152.pth',
    'C:/srain/model_179.pth',
    'C:/srain/model_188.pth',
    'C:/srain/model_226.pth',
    'C:/srain/model_240.pth'
    # Add more model paths as needed
]

# Image file path
image_path = 'C:/16_Rain_l.jpg'

# Output directory
output_dir = 'test_output_images-srl'
os.makedirs(output_dir, exist_ok=True)

# Load the image
clear_image = Image.open('C:/16_NoFault.jpg')

# Iterate through models and test on the same image
for model_path in model_paths:
    # Instantiate the model
    model = Srcnn()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the noisy image
    noisy_image = Image.open(image_path)
    noisy_tensor = transform(noisy_image)

    # Perform denoising using the model
    with torch.no_grad():
        output = model(noisy_tensor.unsqueeze(0))  # Add batch dimension
    out_norm = (output - output.min()) / (output.max() - output.min())
    denoised_image = image_transform(out_norm.squeeze())  # Remove batch dimension and convert to PIL Image

    # Create a composite image
    composite = Image.new('RGB', (clear_image.width * 3, clear_image.height))
    composite.paste(clear_image, (0, 0))  # Clear image on the left
    composite.paste(noisy_image, (clear_image.width, 0))  # Noisy image in the middle
    composite.paste(denoised_image, (clear_image.width * 2, 0))  # Denoised image on the right


    # Save the composite image
    output_filename = f'composite_autoencoder_{os.path.basename(model_path)}_{os.path.basename(image_path)}'
    composite.save(os.path.join(output_dir, output_filename))

    # Convert PIL Image objects to NumPy arrays
    noisy_np = np.array(noisy_image)
    denoised_np = np.array(denoised_image)

    # Calculate metrics (you can add your metric calculation here if needed)

    # Print or save the metrics as needed
    print(f'Model: Autoencoders, Model Path: {model_path}, Image: {image_path}')
    print('---')

print(f'Composite images saved to {output_dir}')


