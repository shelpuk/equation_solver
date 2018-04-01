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
from loss_function import eq_nll_loss
import sys

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
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = (F.relu(self.fc1_bn(self.fc1(x))))
        x = (F.relu(self.fc2_bn(self.fc2(x))))
        x = (F.relu(self.fc3_bn(self.fc3(x))))
        x = F.sigmoid(self.fc4(x))
        return x

model = Net()

model.cuda()

#model = torch.load('test.pt')

f = open('log_weight.log', 'w')

#if torch.cuda.device_count() > 1:
#  print("Let's use", torch.cuda.device_count(), "GPUs!")
#  model = nn.DataParallel(model)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched['feature_vector']
        label = sample_batched['label']
        label_array = sample_batched['label_array']
        weight = sample_batched['weight']

        data, label, label_array, weight = data.cuda(), label.cuda(), label_array.cuda(), weight.cuda()
        data, label, label_array, weight = Variable(data), Variable(label), Variable(label_array), Variable(weight).float()
        optimizer.zero_grad()
        output = model(data)
        loss = eq_nll_loss(input=output, target=label, eq_weight=weight.long())
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            event = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), float(loss) / train_loader.batch_size)
            print(event)
            f.write(event+'\n')

def test():
    model.eval()
    test_loss = 0
    correct = 0

    for batch_idx, sample_batched in enumerate(test_loader):
        data = sample_batched['feature_vector']
        label = sample_batched['label']
        label_array = sample_batched['label_array']
        weight = sample_batched['weight']

        data, label, label_array, weight = data.cuda(), label.cuda(), label_array.cuda(), weight.cuda()
        data, label, label_array, weight = Variable(data), Variable(label), Variable(label_array), Variable(weight).float()

        output = model(data)

        test_loss += eq_nll_loss(input=output, target=label, eq_weight=weight)  # sum up batch loss
        #pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        pred = torch.round(output)

        #print(pred)
        #print(label)

        #correct += sum(pred.long() == label).cpu()

        correct += pred.eq(label.float()).long().cpu().sum()

    #test_loss /= len(test_loader.dataset)
    event = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(float(test_loss)/len(test_loader.dataset), int(correct), len(test_loader.dataset), 100. * int(correct) / len(test_loader.dataset))
    print(event)
    f.write(event+'\n')

for epoch in range(1, 1000):
    test()
    train(epoch)
    torch.save(model, 'test1.pt')
