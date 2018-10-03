import torch.nn as nn
import torch.nn.functional as F

class eq_net_ff(nn.Module):
    def __init__(self):
        super(eq_net_ff, self).__init__()
        n = 5
        self.fc1 = nn.Linear(34,n)
        self.fc1_bn = nn.BatchNorm1d(n)
        self.fc2 = nn.Linear(n, n)
        self.fc2_bn = nn.BatchNorm1d(n)
        # self.fc3 = nn.Linear(n, n)
        # self.fc3_bn = nn.BatchNorm1d(n)
        # self.fc4 = nn.Linear(n, n)
        # self.fc4_bn = nn.BatchNorm1d(n)
        # self.fc5 = nn.Linear(n, n)
        # self.fc5_bn = nn.BatchNorm1d(n)
        # self.fc6 = nn.Linear(n, n)
        # self.fc6_bn = nn.BatchNorm1d(n)
        # self.fc7 = nn.Linear(n, n)
        # self.fc7_bn = nn.BatchNorm1d(n)
        # self.fc8 = nn.Linear(n, n)
        # self.fc8_bn = nn.BatchNorm1d(n)
        # self.fc9 = nn.Linear(n, n)
        # self.fc9_bn = nn.BatchNorm1d(n)
        # self.fc10 = nn.Linear(n, n)
        # self.fc10_bn = nn.BatchNorm1d(n)
        self.fc3 = nn.Linear(n, 1)

    def forward(self, x):
        x = (F.relu(self.fc1_bn(self.fc1(x))))
        x = (F.relu(self.fc2_bn(self.fc2(x))))
        # x = (F.relu(self.fc3_bn(self.fc3(x))))
        # x = (F.relu(self.fc4_bn(self.fc4(x))))
        # x = (F.relu(self.fc5_bn(self.fc5(x))))
        # x = (F.relu(self.fc6_bn(self.fc6(x))))
        # x = (F.relu(self.fc7_bn(self.fc7(x))))
        # x = (F.relu(self.fc8_bn(self.fc8(x))))
        # x = (F.relu(self.fc9_bn(self.fc9(x))))
        # x = (F.relu(self.fc10_bn(self.fc10(x))))
        x = F.sigmoid(self.fc3(x))
        return x