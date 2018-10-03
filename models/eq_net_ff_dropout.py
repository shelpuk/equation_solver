import torch.nn as nn
import torch.nn.functional as F

class eq_net_ff_dropout(nn.Module):
    def __init__(self):
        super(eq_net_ff_dropout, self).__init__()
        # self.fc1 = nn.Linear(32, 20)
        # self.fc1_bn = nn.BatchNorm1d(20)
        # self.fc2 = nn.Linear(20, 20)
        # self.fc2_bn = nn.BatchNorm1d(20)
        # self.fc2_dropout = nn.Dropout()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        # x = (F.relu(self.fc1_bn(self.fc1(x))))
        # x = self.fc2_dropout((F.relu(self.fc2_bn(self.fc2(x)))))
        x = F.sigmoid(self.output(x))
        return x