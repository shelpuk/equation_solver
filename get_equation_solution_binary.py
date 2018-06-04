from dataset_generator import *
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def binary_to_int(binary_array):
    return int(str(binary_array), 2)

def dump_solutions(equation_a, equation_b, true_x, solutions, path_to_folder):
    equation_a_array = equation_a.data.numpy()
    equation_b_array = equation_b.data.numpy()
    solution_array = solutions.data.numpy()

    solution_file = open('./'+ path_to_folder + '/' + 'solutions.csv', 'w')
    solution_file.write('a, b, true_x, suggested_x\n')

    for i in range(equation_a_array.shape[0]):
        a = binary_to_int(equation_a_array[i,:])
        b = binary_to_int(equation_b_array[i,:])
        x = binary_to_int(solution_array[i,:])

        solution_file.write(str(a) + ',' + str(b) + ',' + str(true_x[i]) + ',' + str(x) + '\n')



test_dataset = equation_binary_dataset_cv(cv_set_file='cv_set_linear.p', right_asnwer_chance=.5)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=16)

path_to_folder = './eq_net_ff_dropout_04-10-22:28/'  # change here directory name
the_model = torch.load(path_to_folder + 'model.pt')
print(repr(the_model))
weights = the_model.state_dict()
the_model = the_model.cuda()

correct = []
batch_size = 50

for batch_idx, sample_batched in enumerate(test_loader):

    data = sample_batched['feature_vector'].numpy()
    true_x = sample_batched['true_x']

    equation_a = Variable(torch.from_numpy(data[:, 0:10]), requires_grad=False)
    equation_b = Variable(torch.from_numpy(data[:, 20:40]), requires_grad=False)

    answer = Variable(torch.zeros(50, 10), requires_grad=True)

    #optimizer = optim.SGD([answer], lr=0.0001, momentum=0.5)
    optimizer = optim.Adam([answer], lr=0.001)

    optimizer.zero_grad()

    #print(equation.size())
    #print(answer.size())

    #input_data = torch.cat((equation.float(), answer.float()), 1)

    #print(input_data.size())

    for i in range(1000000):
        input_data = torch.cat((equation_a.float(), F.sigmoid(answer.float()), equation_b.float()), 1)
        output = the_model(input_data.cuda())
        #loss = torch.sum(1. - output)
        loss = torch.sum(1. - output)
        if i % 1000 == 0:
            print('Iter: ', str(i), ', loss: ', float(loss))

        if i % 1000 == 0:
            print(F.sigmoid(answer[:3,:]))
            #dump_solutions(equation_a=equation_a,
            #               equation_b=equation_b,
            #               solutions=answer,
            #               true_x=true_x,
            #               path_to_folder=path_to_folder)

        optimizer.zero_grad()
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