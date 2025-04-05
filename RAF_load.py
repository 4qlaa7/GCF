import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = load_image(image_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Function to load image and resize
def load_image(file_path, size=(224, 224)):
    img = Image.open(file_path)
    img = img.resize(size)
    return np.array(img)

# Path to the folder containing images
image_folder = "D:/GCF/adf/basic/Image/original"

# Path to the text file containing image names and labels
label_file = "D:/GCF/adf/basic/EmoLabel/list_patition_label.txt"

# Dictionary to store image names and labels
image_labels = {}

# Read image names and labels from the text file
with open(label_file, 'r') as file:
    for line in file:
        image_name, label = line.strip().split()
        image_labels[image_name] = label

# Split data into train, test, and validation sets
train_data = []
test_data = []
validation_data = []
for image_name, label in image_labels.items():
    if "train" in image_name:
        if random.random() < 0.8:  # 80% for training
            train_data.append((image_name, label))
        else:  # 20% for validation
            validation_data.append((image_name, label))
    elif "test" in image_name:
        test_data.append((image_name, label))

# Function to load data
def load_data(data):
    images = []
    labels = []
    for image_name, label in data:
        image_path = os.path.join(image_folder, image_name)
        images.append(image_path)
        labels.append(int(label))  # Convert label to integer if needed
    return images, labels

train_images, train_labels = load_data(train_data)
test_images, test_labels = load_data(test_data)
validation_images, validation_labels = load_data(validation_data)

# Define transformations for data augmentation or normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add more transformations as needed
])

# Create custom datasets
train_dataset = CustomDataset(train_images, train_labels, transform=transform)
test_dataset = CustomDataset(test_images, test_labels, transform=transform)
val_dataset = CustomDataset(validation_images, validation_labels, transform=transform)

# Define batch size
batch_size = 256

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Now you have your train_loader, test_loader, and val_loader ready to be used for training, testing, and validation!
print("Train images :",len(train_images))
print("Test images :",len(test_images))
print("validation images :",len(validation_images))