#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH -t 24:00:00
#SBATCH --mem=50G
python /home/g051226/autoencoder/main.py
