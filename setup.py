import os
import requests
from zipfile import ZipFile
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Install necessary libraries
!pip install torch torchvision torchaudio opencv-python numpy scikit-learn matplotlib mediapipe

print("Environment setup complete!")