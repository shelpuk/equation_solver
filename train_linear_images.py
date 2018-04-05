from dataset_generator_images import *
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from loss_function import eq_nll_loss
from models import *
import models

train_dataset = equation_linear_images_dataset_train(cv_set_file='cv_set_linear.p', right_asnwer_chance=.5)
test_dataset = equation_linear_images_dataset_cv(cv_set_file='cv_set_linear.p', right_asnwer_chance=.5)

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=16)

#for i_batch, sample_batched in enumerate(dataloader):
#    print(sample_batched['feature_vector'].shape)
#    if i_batch == 3: break

model = models.LeNet()

model.cuda()

#model = torch.load('test.pt')

f = open('log_linear_images_lenet.log', 'w')

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
        output = model(data.float())
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

    file_error_analysis = open('errors.log', 'w')
    file_error_analysis.write('a, b, true_x, suggested_x, true_label, prediction\n')

    for batch_idx, sample_batched in enumerate(test_loader):
        data = sample_batched['feature_vector']
        label = sample_batched['label']
        label_array = sample_batched['label_array']
        weight = sample_batched['weight']
        a = sample_batched['a']
        b = sample_batched['b']
        true_x = sample_batched['true_x']
        x = sample_batched['x']

        data, label, label_array, weight = data.cuda(), label.cuda(), label_array.cuda(), weight.cuda()
        data, label, label_array, weight = Variable(data), Variable(label), Variable(label_array), Variable(weight).float()

        output = model(data.float())

        test_loss += eq_nll_loss(input=output, target=label, eq_weight=weight)  # sum up batch loss
        #pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        pred = torch.round(output)

        #print(pred)
        #print(label)

        #correct += sum(pred.long() == label).cpu()

        correct += pred.eq(label.float()).long().cpu().sum()

        # Error analysis

        #print(label.data.cpu())
        #output = output.data.cpu().numpy()

        match = pred.eq(label.float()).long().cpu()
        error_ids = (match == 0).nonzero()[:,0]
        for i in list(error_ids):
            i = int(i)
            file_error_analysis.write(str(a.numpy()[i])+','
                                      +str(b.numpy()[i])+','
                                      +str(true_x.numpy()[i])+','
                                      +str(x.numpy()[i])+','
                                      +str(label.data.cpu().numpy()[i][0])+','
                                      +str(output.data.cpu().numpy()[i][0])+'\n')

    #test_loss /= len(test_loader.dataset)

    event = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(float(test_loss)/len(test_loader.dataset), int(correct), len(test_loader.dataset), 100. * int(correct) / len(test_loader.dataset))
    print(event)
    f.write(event+'\n')
    file_error_analysis.close()


for epoch in range(1, 1000):
    test()
    train(epoch)
    torch.save(model, 'test1.pt')
