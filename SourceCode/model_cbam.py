
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropout import get_dropout
import config
from cbam import CBAMBlock

class CNN_CBAM(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(CNN_CBAM, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = get_dropout()
        self.cbam = CBAMBlock(64)
        self.fc1 = nn.Linear(64*12*12,128)
        self.fc2 = nn.Linear(128,num_classes)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.cbam(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__=="__main__":
    model = CNN_CBAM()
    x = torch.randn(4,1,48,48)
    out = model(x)
    print(out.shape) 