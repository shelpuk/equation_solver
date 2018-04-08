import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 12, 5)
        self.conv1_bn = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.fc1   = nn.Linear(2816, 200)
        self.fc1_bn = nn.BatchNorm1d(200)
        self.fc1_dropout = nn.Dropout()
        self.fc2   = nn.Linear(200, 100)
        self.fc2_bn = nn.BatchNorm1d(100)
        self.fc2_dropout = nn.Dropout()
        self.fc3   = nn.Linear(100, 1)

    def forward(self, x):
        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1_dropout(F.relu(self.fc1_bn(self.fc1(out))))
        out = self.fc2_dropout(F.relu(self.fc2_bn(self.fc2(out))))
        out = F.sigmoid(self.fc3(out))

        return out