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
    'C:/low/srain/model_0.pth',
    'C:/low/srain/model_1.pth',
    'C:/low/srain/model_3.pth',
    'C:/low/srain/model_5.pth',
    'C:/low/srain/model_7.pth',
    'C:/low/srain/model_9.pth',
    'C:/low/srain/model_20.pth',
    'C:/low/srain/model_24.pth',
    'C:/low/srain/model_35.pth',
    'C:/low/srain/model_49.pth',
    'C:/low/srain/model_59.pth',
    'C:/low/srain/model_71.pth',
    'C:/low/srain/model_84.pth',
    'C:/low/srain/model_103.pth',
    'C:/low/srain/model_134.pth',
    'C:/low/srain/model_147.pth',
    'C:/low/srain/model_156.pth',
    'C:/low/srain/model_190.pth',
    'C:/low/srain/model_227.pth'
    # Add more model paths as needed
]

# Image file path
image_path = 'C:/16_Rain_l.jpg'

# Output directory
output_dir = 'test_output_images-srl'
os.makedirs(output_dir, exist_ok=True)

# Load the image
image_path1 = 'C:/16_NoFault.jpg'
metrics_data = []

# Iterate through models and test on the same image
for model_path in model_paths:
    # Instantiate the model
    model = Srcnn()
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

print(f'Metrics data saved to {output_dir}')



    # Print or save the metrics as needed
print(f'Model: Autoencoders, Model Path: {model_path}, Image: {image_path}')
print('---')

print(f'Composite images saved to {output_dir}')

