from dataset_generator_images import *
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim


def convert_to_image(img_array):
    min = np.min(img_array)
    max = np.max(img_array)
    if min < 0:
        img_array = 1.*(img_array - min) / ((1.*max / (max - min)))

    img_array = img_array * 255
    img_array = np.clip(img_array, a_min=0, a_max=255).astype(np.uint8)

    img = Image.fromarray(img_array)

    img.show()

    return img

test_dataset = equation_linear_images_dataset_cv(cv_set_file='cv_set_linear.p', right_asnwer_chance=.5)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=16)

path_to_folder = './LeNet_reduced_dropout_04-08-23:59/'  # change here directory name
the_model = torch.load(path_to_folder + 'model.pt')
print(repr(the_model))
weights = the_model.state_dict()
the_model = the_model.cuda()

correct = []
batch_size = 50

for batch_idx, sample_batched in enumerate(test_loader):
    data = sample_batched['feature_vector'].numpy()[:,0,:,:].reshape((50,1,30,100))

    #print(data.shape)

    #print(data[:,:,:,:].size())

    #equation = Variable(data[0,0,:,:].view(1, 1, 30, 100), requires_grad=False)
    equation = Variable(torch.from_numpy(data), requires_grad=False)
    #img = convert_to_image(img_array)

    answer = Variable(torch.randn(50, 1, 30, 100), requires_grad=True)

    optimizer = optim.SGD([answer], lr=0.01, momentum=0.5)
    optimizer.zero_grad()

    #print(equation.size())
    #print(answer.size())

    #input_data = torch.cat((equation.float(), answer.float()), 1)

    #print(input_data.size())

    for i in range(1000000):
        input_data = torch.cat((equation.float(), answer.float()), 1)
        output = the_model(input_data.cuda())
        loss = torch.sum(1. - output)
        if i % 1000 == 0:
            print('Iter: ', str(i), ', loss: ', float(loss))
        loss.backward()
        optimizer.step()

    #

    #print(input_data.size())

    break

'''
    label = sample_batched['label']
    label_array = sample_batched['label_array']
    weight = sample_batched['weight']
    a = sample_batched['a']
    b = sample_batched['b']
    true_x = sample_batched['true_x']
    x = sample_batched['x']

    data, label, label_array, weight = data.cuda(), label.cuda(), label_array.cuda(), weight.cuda()
    data, label, label_array, weight = Variable(data), Variable(label), Variable(label_array), Variable(
        weight).float()

    output = the_model(data.float())
    pred = torch.round(output)
    correct_ = sum(pred.long() == label).cpu().data.numpy()[0] / batch_size
    correct.append(correct_)
    # correct += pred.eq(label.float()).long().cpu().sum()

print("Average accuracy: {}".format(np.mean(correct)))
'''