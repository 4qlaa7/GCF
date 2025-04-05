import torch
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

def Read_Data(path,label_mapping,device, images_per_class=-1, test_size = 0.2, val_size = 0.1, batch_size = 90):
 
  all_images = []
  all_labels = []
  
    
  for class_folder in os.listdir(path):
      class_path = os.path.join(path, class_folder)

      if os.path.isdir(class_path):
          class_label = label_mapping.get(class_folder, -1)

          if class_label != -1:
              i=0
              for image_file in os.listdir(class_path):
                  if images_per_class !=-1 and i >= images_per_class:
                    break
                  image_path = os.path.join(class_path, image_file)
                  image_path = os.path.normpath(image_path).replace('\\', '/')
                  #print(image_path)
                  image = cv2.imread(image_path)
                  if image is None:
                    print("Error: Unable to load image at", image_path)
                    print(i)
                    continue
                  
                  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                  #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                  image = cv2.resize(image, (360, 360))
                  #print(image.shape)
                  all_images.append(image)
                  all_labels.append(class_label)
                  #if i < images_per_class:
                    # Apply random affine transformations (rotation, translation, scaling)
                   # affine_transform = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1),scale=(0.9, 1.1))
                                                            
                    #augmented_image = affine_transform(transforms.ToPILImage()(image))
                    #image = transforms.ToTensor()(augmented_image)
                    #all_images.append(image)
                    #all_labels.append(class_label)
                  i+=1
              print (class_label, images_per_class, i)

  all_images = torch.stack([torch.tensor(image).permute(2,0,1) for image in all_images])
  all_labels = torch.tensor(all_labels)
  print(all_images.shape)
  train_images, temp_images, train_labels, temp_labels = train_test_split(
      all_images, all_labels, test_size=test_size + val_size, random_state=42
  )

  test_images, val_images, test_labels, val_labels = train_test_split(
      temp_images, temp_labels, test_size=val_size / (test_size + val_size), random_state=42
  )

  transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

  # Create datasets and dataloaders
  class CustomDataset(Dataset):
      def __init__(self, images, labels, transform=None):
          self.images = images
          self.labels = labels
          self.transform = transform

      def __len__(self):
          return len(self.images)

      def __getitem__(self, idx):
          image = self.images[idx]
          label = self.labels[idx]

          # Convert image to tensor if it's not already
          if not isinstance(image, torch.Tensor):
              image = torch.tensor(image, dtype=torch.float32)

          if self.transform:
              # If image is a tensor, apply the transformation directly
              if isinstance(image, torch.Tensor):
                  image = self.transform(image)
              else:
                  # If image is still a PIL Image, apply ToTensor transformation
                  image = self.transform(transforms.ToPILImage()(image))

          return image, label

  # transform=transforms.Compose([
  #   transforms.ToPILImage(),
  #   transforms.ToTensor(),
  #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  # ])

  train_dataset = CustomDataset(train_images, train_labels, transform=transform)
  test_dataset = CustomDataset(test_images, test_labels, transform=transform)
  val_dataset = CustomDataset(val_images, val_labels, transform=transform)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

  return train_loader,test_loader,val_loader