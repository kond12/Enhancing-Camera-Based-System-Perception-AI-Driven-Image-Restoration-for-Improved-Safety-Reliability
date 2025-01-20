import os
import matplotlib.pyplot as plt
import pandas as pd
import csv

plot_folder = "plots_folder"

# Create the folder if it doesn't exist
os.makedirs(plot_folder, exist_ok=True)

def plot_metrics_and_loss(epochs, train_losses, valid_losses, psnr_list, ssim_list, mse_list, mae_list):
     
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label="Training loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    
    # Save the plot to the folder
    plot_filename = os.path.join(plot_folder, "Training_loss_plot.png")
    plt.savefig(plot_filename)
    
    # Close the plot to free up resources (optional)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), valid_losses, label="Validation loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Losses")
    
    # Save the plot to the folder
    plot_filename = os.path.join(plot_folder, "Validation_loss_plot.png")
    plt.savefig(plot_filename)
    
    # Close the plot to free up resources (optional)
    plt.close()
    
    # Plot the training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label="Training loss")
    plt.plot(range(epochs), valid_losses, label="Validation loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    
    # Save the plot to the folder
    plot_filename = os.path.join(plot_folder, "loss_plot.png")
    plt.savefig(plot_filename)
    
    # Close the plot to free up resources (optional)
    plt.close()

    # Plot PSNR
    plt.figure(figsize=(10, 6))
    #plt.plot(range(1, epochs + 1), psnr_list, label="PSNR")
    
    #plt.plot(range(1, epochs + 1)[:len(psnr_list)], psnr_list, label="PSNR")
    plt.plot(range(epochs), psnr_list, label="PSNR")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("PSNR")

    plt.title("PSNR")

    # Save the plot to the folder
    plot_filename = os.path.join(plot_folder, "psnr_plot.png")
    plt.savefig(plot_filename)
    
    # Close the plot to free up resources (optional)
    plt.close()

    # Plot SSIM
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), ssim_list, label="SSIM")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("SSIM")
    plt.title("SSIM")

    # Save the plot to the folder
    plot_filename = os.path.join(plot_folder, "ssim_plot.png")
    plt.savefig(plot_filename)
    
    # Close the plot to free up resources (optional)
    plt.close()

    # Plot MSE
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), mse_list, label="MSE")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("MSE")

    # Save the plot to the folder
    plot_filename = os.path.join(plot_folder, "mse_plot.png")
    plt.savefig(plot_filename)
    
    # Close the plot to free up resources (optional)
    plt.close()

    # Plot MAE
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), mae_list, label="MAE")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.title("MAE")

    # Save the plot to the folder
    plot_filename = os.path.join(plot_folder, "mae_plot.png")
    plt.savefig(plot_filename)
    
    # Close the plot to free up resources (optional)
    plt.close()

    # Combined Metrics Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), psnr_list, label="PSNR")
    plt.plot(range(epochs), ssim_list, label="SSIM")
    plt.plot(range(epochs), mse_list, label="MSE")
    plt.plot(range(epochs), mae_list, label="MAE")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Combined Metrics (PSNR, SSIM, MSE, MAE)")

    # Save the plot to the folder
    plot_filename = os.path.join(plot_folder, "combined_metrics_plot.png")
    plt.savefig(plot_filename)
    
    # Close the plot to free up resources (optional)
    plt.close()


#csv_filename = "metrics_and_losses.csv"
def save_metrics_and_losses_to_csv(epochs, train_losses, valid_losses, psnr_list, ssim_list, mse_list, mae_list, csv_filename):
    
    data = {
        'Epoch': list(range(epochs)),
        'Train Loss': train_losses,
        'Valid Loss': valid_losses,
        'PSNR': psnr_list,
        'SSIM': ssim_list,
        'MSE': mse_list,
        'MAE': mae_list
    }

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for epoch in range(epochs):
            writer.writerow({
                'Epoch': epoch,
                'Train Loss': train_losses[epoch],
                'Valid Loss': valid_losses[epoch],
                'PSNR': psnr_list[epoch],
                'SSIM': ssim_list[epoch],
                'MSE': mse_list[epoch],
                'MAE': mae_list[epoch]
            })

