import torch
import torch.nn as nn
import torch.nn.functional as F


class VDCNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes, k_max):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings = num_embeddings, 
                                      embedding_dim = embedding_dim, padding_idx = 0)
        
        self.conv = nn.Conv1d(in_channels = embedding_dim, out_channels = 64, kernel_size = 3, 
                              stride = 1, padding =1 )
        
        self.conv_layers = nn.Sequential(
            ResidualBlock(64, 64, True),
            ResidualBlock(64, 64, True, "VGG-like"),
            ResidualBlock(64, 128, True),
            ResidualBlock(128, 128, True, "VGG-like"),
            ResidualBlock(128, 256, True),
            ResidualBlock(256, 256, True, "VGG-like"),
            ResidualBlock(256, 512, True),
            ResidualBlock(512, 512)
        )
        
        self.k_max_pooling = nn.AdaptiveMaxPool1d(8)
        
        self.flatten = Flatten()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 8, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )
        
        self.softmax = nn.LogSoftmax(dim = 1)
       
    def forward(self, x):
        x = self.embedding(x) 
        x = x.transpose(1, 2) 
        x = self.conv(x) 
        x = self.conv_layers(x)
        x = self.k_max_pooling(x)
        x = self.flatten(x)
        x = self.fc_layers(x) 
        return x

class ConvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, conv_1st_stride):
        super().__init__()
        self.conv_1st = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=conv_1st_stride, padding=1)
        self.conv_2nd = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_1st(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv_2nd(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input:torch.tensor):
        return torch.flatten(input, start_dim=1)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut = None, down_sampling = None):
        super().__init__()
        self.shortcut = shortcut
        self.down_sampling = down_sampling
        
        if self.down_sampling == "Resnet-like":
            conv_1st_stride = 2 
        else:
            conv_1st_stride = 1
        
        self.conv_block = ConvBlock(in_channels, out_channels, conv_1st_stride)
        
        if self.down_sampling == "VGG-like":
            self.max_pooling = nn.MaxPool1d(3, 2, padding = 1)
        
        if self.down_sampling:
            shortcut_conv_stride = 2
        else:
            shortcut_conv_stride = 1
        
        self.shortcut_conv = nn.Conv1d(in_channels, out_channels, 1, shortcut_conv_stride)

    def forward(self, x):
        y = self.conv_block(x)
        if self.down_sampling == "VGG-like":
            y = self.max_pooling(y)
        if self.shortcut:
            y += self.shortcut_conv(x) 
        return y