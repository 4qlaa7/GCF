import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch_geometric.nn as geom_nn


class GCF(nn.Module):
    def __init__(self, num_classes, device, base_model='vgg16', mode='cnn+gcn'):
        super(GCF, self).__init__()
        self.mode = mode
        self.device = device
        self.base_model = base_model.lower()

        # Load the selected pretrained model and define feature size
        cnn_output_size = 0

        if self.base_model == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1], nn.Flatten())
            cnn_output_size = 4096

        elif self.base_model == 'vgg19':
            self.model = models.vgg19(pretrained=True)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1], nn.Flatten())
            cnn_output_size = 4096

        elif self.base_model == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Flatten()
            cnn_output_size = 512

        elif self.base_model == 'resnet34':
            self.model = models.resnet34(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Flatten()
            cnn_output_size = 512

        elif self.base_model == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Flatten()
            cnn_output_size = 2048

        elif self.base_model == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(nn.Flatten())
            cnn_output_size = 1024

        elif self.base_model == 'densenet169':
            self.model = models.densenet169(pretrained=True)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(nn.Flatten())
            cnn_output_size = 1664

        elif self.base_model == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(nn.Flatten())
            cnn_output_size = 1920

        elif self.base_model == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Flatten()
            cnn_output_size = 1280

        elif self.base_model == 'inception_v3':
            self.model = models.inception_v3(pretrained=True, aux_logits=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Flatten()
            cnn_output_size = 2048

        else:
            raise ValueError(f"Unsupported base model: {self.base_model}")

        # Define output sizes and fully connected layers
        gcn_output_size = 128
        if self.mode == "cnn" or self.mode == "cnn+gcn":
            self.fc_cnn = nn.Linear(num_ftrs, cnn_output_size)
        if self.mode == "gcn" or self.mode == "cnn+gcn":
            self.fc_gcn = nn.Linear(256, gcn_output_size)

        combined_output_size = cnn_output_size + gcn_output_size if mode == "cnn+gcn" else max(cnn_output_size, gcn_output_size)
        self.fc = nn.Linear(combined_output_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.5)
        self.gnn = self._build_gnn_layer(cnn_output_size)

    def _build_gnn_layer(self, cnn_output_size):
        return GNNLayer(in_channels=cnn_output_size, hidden_channels=256, out_channels=256, device=self.device)

    def forward(self, images):
        images = images.to(self.device)
        cnn_features = self.model(images)

        if self.mode == "cnn" or self.mode == "cnn+gcn":
            cnn_features = cnn_features.view(cnn_features.size(0), -1)
            cnn_features = self.fc_cnn(cnn_features)

        if self.mode == "gcn" or self.mode == "cnn+gcn":
            gcn_features = self.gnn(cnn_features)
            gcn_features = gcn_features.view(gcn_features.size(0), -1)
            gcn_features = self.fc_gcn(gcn_features)

        if self.mode == "cnn+gcn":
            features = torch.cat((cnn_features, gcn_features), dim=1)
        elif self.mode == "cnn":
            features = cnn_features
        else:
            features = gcn_features

        out = self.fc(features)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return out

class GNNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super(GNNLayer, self).__init__()
        self.conv1 = geom_nn.GraphConv(in_channels, hidden_channels)
        self.device = device
        
        
        #VERSION 2
        #self.edge_index = torch.tensor(
        #    [[0, 1, 0, 3, 0, 4, 1, 2, 1, 4, 3, 4, 3, 6, 4, 5, 4, 7, 6, 7, 7, 8, 2, 4, 4, 6, 5, 4, 4, 8],
        #     [0, 1, 0, 3, 0, 4, 1, 2, 1, 4, 3, 4, 3, 6, 4, 5, 4, 7, 6, 7, 7, 8, 2, 4, 4, 6, 5, 4, 4, 8]],
        #    dtype=torch.long).to(self.device)
        

        # VERSION 3
        self.edge_index = torch.tensor(
            [[1, 0, 3, 0, 4, 0, 2, 1, 4, 1, 4, 3, 6, 3, 5, 4, 7, 4, 7, 6, 8, 7, 4, 2, 6, 4, 4, 5, 8, 4],
             [1, 0, 3, 0, 4, 0, 2, 1, 4, 1, 4, 3, 6, 3, 5, 4, 7, 4, 7, 6, 8, 7, 4, 2, 6, 4, 4, 5, 8, 4]],
            dtype=torch.long).to(self.device)

    def forward(self, x):
        return F.relu(self.conv1(x, self.edge_index))