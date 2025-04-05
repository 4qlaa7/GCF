# GCF (Graph Convolutional Features) Model Repository

## Overview
This repository contains the implementation of a deep learning pipeline utilizing a Graph Convolutional Features (GCF) Model combined with traditional CNN architectures (e.g., VGG, ResNet, DenseNet, EfficientNet, and Inception). The code is structured to facilitate training, validation, and testing of image classification tasks enhanced by graph-based methods.

## Project Structure
- `Data_Loader.py`: Handles loading, preprocessing, splitting, and batching the dataset.
- `RAF_load.py`: Custom dataset loader specifically designed for datasets with images listed in text files.
- `GCF_Model.py`: Defines the architecture of the Graph Convolutional Features Model, combining CNNs and Graph Convolutional Networks (GCNs).
- `Training.py`: Contains the training and validation routines.
- `server_training.ipynb`: Jupyter notebook demonstrating usage and training procedure.

## Features
- Supports various CNN backbone models (`vgg16`, `resnet50`, `densenet121`, etc.).
- Flexible integration of CNN and GCN (`cnn`, `gcn`, `cnn+gcn` modes).
- Customizable dataset handling with support for train/test/validation splits.
- Built-in data augmentation and normalization.

## Requirements
- Python 3.x
- PyTorch
- Torchvision
- PyTorch Geometric
- OpenCV
- NumPy
- Scikit-learn
- Pillow

## Installation
Clone the repository:
```bash
git clone https://github.com/4qlaa7/GCF.git
cd GCF
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Data Preparation
Update the paths and configurations within `Data_Loader.py` or `RAF_load.py` according to your dataset.

### Model Training
Use the `Training.py` script or `server_training.ipynb` notebook for training and evaluating the model.

Example training command:
```bash
python Training.py
```

### Configurations
Modify parameters such as:
- Model type (`base_model`)
- Training epochs (`num_epochs`)
- Batch size (`batch_size`)

in the respective files according to your experimentation needs.

## Contributions
Feel free to fork the repository, create pull requests, or open issues for enhancements, bug reports, or questions.

## License
Distributed under the MIT License. See `LICENSE` for more information.

---
Happy Coding!

