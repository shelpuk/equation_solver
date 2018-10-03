from dataset_generator import *
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from loss_function import eq_nll_loss
from models import *
import models
import os, sys
import time
import torch.nn

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bce_loss = torch.nn.BCELoss()

#mse_loss = torch.nn.MSELoss()

batch_size = 50

train_dataset = equation_binary_dataset_train_4(cv_set_file='cv_set_4.p', right_answer_chance=.5)
test_dataset = equation_binary_dataset_cv_4(cv_set_file='cv_set_4.p', right_answer_chance=.5)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# for i_batch, sample_batched in enumerate(dataloader):
#    print(sample_batched['feature_vector'].shape)
#    if i_batch == 3: break

model = models.eq_net_ff()
#model.cuda()
summary = repr(model)
# model = torch.load('test.pt')
print(summary)

# path_to_folder = './eq_net_ff_08-30-11-54/'  # change here directory name
# model = torch.load(path_to_folder + 'model.pt')
# summary = repr(model)
# print(repr(model))
# weights = model.state_dict()

mod_name = summary.split('\n')[0][:-1]  # model name
dir = mod_name + "_" + time.strftime("%m-%d-%H-%M")
print(dir)
if not os.path.exists(dir):
    os.makedirs(dir)

f = open('./'+ dir + '/' + 'log_' + mod_name + '.log', 'w')
f.write(summary + '\n')

# if torch.cuda.device_count() > 1:
#  print("Let's use", torch.cuda.device_count(), "GPUs!")
#  model = torch.nn.DataParallel(model)

#optimizer = optim.RMSprop(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched['feature_vector']
        #print(data[0].shape)
        #print(data)
        label = sample_batched['label']
        label_array = sample_batched['label_array']
        weight = sample_batched['weight']

#        data, label, label_array, weight = data.cuda(), label.cuda(), label_array.cuda(), weight.cuda()
        data, label, label_array, weight = Variable(data), Variable(label), Variable(label_array), Variable(
            weight).float()
        optimizer.zero_grad()
        output = model(data.float())
        # print(output.detach().numpy().shape)
        loss =bce_loss(input=output, target=label.float())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            event = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                             len(train_loader.dataset),
                                                                             100. * batch_idx / len(train_loader),
                                                                             float(loss) / train_loader.batch_size)
            print(event)
            f.write(event + '\n')


def test():
    model.eval()
    test_loss = 0
    correct = 0

    file_error_analysis = open('./'+ dir + '/' + 'errors.csv', 'w')
    file_full_activation = open('./' + dir + '/' + 'activations.csv', 'w')

    file_error_analysis.write('a, b, true_x, suggested_x, true_label, prediction\n')
    file_full_activation.write('a, b, true_x, suggested_x, true_label, prediction, correct\n')

    for batch_idx, sample_batched in enumerate(test_loader):
        data = sample_batched['feature_vector']
        # print(data.shape)
        label = sample_batched['label']
        label_array = sample_batched['label_array']
        weight = sample_batched['weight']
        a = sample_batched['a']
        b = sample_batched['b']
        true_x = sample_batched['true_x']
        x = sample_batched['x']

#        data, label, label_array, weight = data.cuda(), label.cuda(), label_array.cuda(), weight.cuda()
        data, label, label_array, weight = Variable(data), Variable(label), Variable(label_array), Variable(
            weight).float()

        output = model(data.float())
        #print(output.shape)
        test_loss += bce_loss(input=output, target=label.float())  # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        pred = torch.round(output)

        # print(pred)
        # print(label)
        # correct += sum(pred.long() == label).cpu()

        correct += pred.eq(label.view(batch_size, 1).float()).long().cpu().sum()
        correct_array = pred.eq(label.float()).long().cpu()

        # Error analysis

        # print (label.data.cpu().numpy().shape)
        # print(output.data.cpu().numpy().shape)
        # print(correct_array.data.cpu().numpy().shape)

        for i in range(len(data)):
            file_full_activation.write(str(a.numpy()[i]) + ','
                                      + str(b.numpy()[i]) + ','
                                      + str(true_x.numpy()[i]) + ','
                                      + str(x.numpy()[i]) + ','
                                      + str(label.data.cpu().numpy()[i][0]) + ','
                                      + str(output.data.cpu().numpy()[i][0]) + ','
                                      + str(correct_array.data.cpu().numpy()[i][0]) +'\n')

        #if int(pred.eq(label.float()).long().cpu().sum()) < test_loader.batch_size:

        match = pred.eq(label.float()).long().cpu()

        if len((match == 0).nonzero().size()) > 1:
            print(len((match == 0).nonzero().size()))
            error_ids = (match == 0).nonzero()[:, 0]
            print(error_ids)
            #print(len(error_ids.size()))
            #print(error_ids.size())

            #if len(error_ids.size()) > 0:

            for i in list(error_ids):
                i = int(i)
                file_error_analysis.write(str(a.numpy()[i]) + ','
                                          + str(b.numpy()[i]) + ','
                                          + str(true_x.numpy()[i]) + ','
                                          + str(x.numpy()[i]) + ','
                                          + str(label.data.cpu().numpy()[i][0]) + ','
                                          + str(output.data.cpu().numpy()[i][0]) + '\n')

    # test_loss /= len(test_loader.dataset)

    event = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(test_loss) / len(test_loader.dataset), int(correct), len(test_loader.dataset),
        100. * int(correct) / len(test_loader.dataset))
    print(event)
    f.write(event + '\n')
    file_error_analysis.close()
    file_full_activation.close()

for epoch in range(1, 2000):
    test()
    train(epoch)
    torch.save(model, './' + dir + '/' + 'model' + '.pt')
