import torch.nn as nn
import torch.nn.functional as F

class eq_net_ff(nn.Module):
    def __init__(self):
        super(eq_net_ff, self).__init__()
        self.fc1 = nn.Linear(40, 200)
        self.fc1_bn = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.fc2_bn = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 200)
        self.fc3_bn = nn.BatchNorm1d(200)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        x = (F.relu(self.fc1_bn(self.fc1(x))))
        x = (F.relu(self.fc2_bn(self.fc2(x))))
        x = (F.relu(self.fc3_bn(self.fc3(x))))
        x = F.sigmoid(self.fc4(x))
        return x