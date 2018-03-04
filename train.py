from dataset_generator import *
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

train_dataset = equation_binary_dataset_train(cv_set_file='cv_set_linear.p', right_asnwer_chance=.5)
test_dataset = equation_binary_dataset_cv(cv_set_file='cv_set_linear.p', right_asnwer_chance=.5)

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=16)

#for i_batch, sample_batched in enumerate(dataloader):
#    print(sample_batched['feature_vector'].shape)
#    if i_batch == 3: break

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(40, 100)
        self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.fc2_bn = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 100)
        self.fc3_bn = nn.BatchNorm1d(100)


    def forward(self, x):
        x = F.dropout(F.relu(self.fc1_bn(self.fc1(x))))
        x = F.dropout(F.relu(self.fc2_bn(self.fc2(x))))
        x = F.dropout(F.relu(self.fc3_bn(self.fc3(x))))
        return F.log_softmax(x, dim=1)

model = Net()

f = open('train2.log', 'w')

#model.load_state_dict(torch.load('test.pt'))

#if torch.cuda.device_count() > 1:
#  print("Let's use", torch.cuda.device_count(), "GPUs!")
#  model = nn.DataParallel(model)

model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched['feature_vector']
        label = sample_batched['label']
        weight = sample_batched['weight']

        data, label, weight = data.cuda(), label.cuda(), weight.cuda()
        data, label, weight = Variable(data), Variable(label), Variable(weight).float()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            event = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0])
            print(event)
            f.write(event+'\n')

def test():
    model.eval()
    test_loss = 0
    correct = 0

    for batch_idx, sample_batched in enumerate(test_loader):
        data = sample_batched['feature_vector']
        label = sample_batched['label']
        weight = sample_batched['weight']

        data, label, weight = data.cuda(), label.cuda(), weight.cuda()
        data, label, weight = Variable(data), Variable(label), Variable(weight).float()

        output = model(data)

        test_loss += F.nll_loss(output, label, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    event = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset))
    print(event)
    f.write(event+'\n')

for epoch in range(1, 1000):
    test()
    train(epoch)
    torch.save(model.state_dict(), 'test1.pt')
