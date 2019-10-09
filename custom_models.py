import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvNet(nn.Module):
    
    def __init__(self, h, w, k, nonlinearity, pad):
        super(ConvNet, self).__init__()
#         self.dd1 = nn.Dropout(p=0.5)
        self.nnl = nonlinearity
        self.conv1 = nn.Conv2d(in_channels=k, out_channels=4, kernel_size=3, padding=pad)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=pad)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=pad)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        
        self.out_size = ((((h - pad) // 2 - pad) // 2 - pad) // 2)       
        self.p = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.nnl(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.nnl(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.nnl(x)
        x = self.p(x)
        x = self.conv4(x)
#         x = x.view(x.size(0), self.out_size ** 2)
#         x = self.nnl(x)
#         x = self.p(x)
        x = x.flatten()
        return x
    

class DenseNet(nn.Module):
    
    def __init__(self, h, w, k):
        super(DenseNet, self).__init__()
        self.dd1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(h*w*k, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.dd2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.dd3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(100, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dd1(x)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.bn1(x)
        
        x = self.dd2(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.bn2(x)
        
        x = self.dd3(x)
        x = self.fc3(x)
        x = x.flatten()
        return x
    
    
class PretrainedNet(nn.Module):
    def __init__(self, model, pretrained):
        super(PretrainedNet, self).__init__()
        resnet = model(pretrained=pretrained)
        modules=list(resnet.children())
        self.pretrained = nn.Sequential(*modules[:-2])
        self.p = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels=modules[-1].in_features, 
                              out_channels=1, 
                              kernel_size=1) 
        
    def forward(self, x):
        x = self.pretrained(x)
        x = self.p(x)
        x = self.conv(x)
        x = x.flatten()
        return x