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
    'C:/low/arain/model_0.pth',
    'C:/low/arain/model_1.pth',
    'C:/low/arain/model_2.pth',
    'C:/low/arain/model_3.pth',
    'C:/low/arain/model_5.pth',
    'C:/low/arain/model_10.pth',
    'C:/low/arain/model_16.pth',
    'C:/low/arain/model_21.pth',
    'C:/low/arain/model_37.pth',
    'C:/low/arain/model_49.pth',
    'C:/low/arain/model_71.pth',
    'C:/low/arain/model_108.pth',
    'C:/low/arain/model_131.pth',
    'C:/low/arain/model_178.pth',
    'C:/low/arain/model_197.pth',
    'C:/low/arain/model_215.pth',
    # Add more model paths as needed
]

# Image file path
image_path = 'C:/16_Rain_l.jpg'

# Output directory
output_dir = 'test_output_imagesarl'
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
