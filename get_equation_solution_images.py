from dataset_generator_images import *
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from multiprocessing import freeze_support
#
# freeze_support()

batch_size = 1

def convert_to_image(img_array):
    min = np.min(img_array)
    max = np.max(img_array)
    if min < 0:
        img_array = 1.*(img_array - min) / ((1.*max / ((max - min)+0.0000001))+0.0000001)

    img_array = img_array * 255
    img_array = np.clip(img_array, a_min=0, a_max=255).astype(np.uint8)

    img = Image.fromarray(img_array)

    #img.show()

    return img_array, img

def dump_solutions(dir, equations, solutions):
    equations_array = equations.data.numpy()
    solutions_array = solutions.data.numpy()

    for i in range(equations_array.shape[0]):
        equation_array, _ = convert_to_image(equations_array[i, 0])
        solution_array, _ = convert_to_image(solutions_array[i, 0])
        stacked_image_array = np.concatenate((equation_array, solution_array), axis=0)
        stacked_image = Image.fromarray(stacked_image_array)
        #stacked_image.show()
        stacked_image.save(dir+'/'+str(i)+'.png')

#test_dataset = equation_linear_images_dataset_cv(cv_set_file='cv_set_linear.p', right_asnwer_chance=1.0)
test_dataset = equation_linear_images_dataset_same_solution(147)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

path_to_folder = './LeNet_reduced_no_bn_07-16-15-13/'  # change here directory name
the_model = torch.load(path_to_folder + 'model.pt')
print(repr(the_model))
weights = the_model.state_dict()
#the_model = the_model.cuda()
the_model.eval()


for batch_idx, sample_batched in enumerate(test_loader):

    data = sample_batched['feature_vector'].numpy()[:,0,:,:].reshape((batch_size,1,30,100))
    # im = Image.fromarray(data.reshape((30,100)))
    # im.show()
    equation = Variable(torch.from_numpy(data), requires_grad=False)

    # Initialize answer as random noise
    #answer = Variable(0.5*torch.rand(50, 1, 30, 100), requires_grad=True)

    # Initialize answer as a black image
    #answer = Variable(torch.ones(batch_size, 1, 30, 100)-1, requires_grad=True)

    #Initialize answer as a right answer
    answer = Variable(torch.from_numpy(sample_batched['feature_vector'].numpy()[:, 1, :, :].reshape((batch_size, 1, 30, 100))), requires_grad=True)

    #optimizer = optim.SGD([answer], lr=0.01, momentum=0.5)
    optimizer = optim.Adam([answer], lr=0.0001)
    #optimizer = optim.RMSprop([answer], lr=0.001)




    optimizer.zero_grad()

    input_data = torch.cat((equation.float(), answer.float()), 1)

    output = the_model(input_data)#.cuda())

    data_test = Variable(torch.from_numpy(sample_batched['feature_vector'].numpy())).float()#.cuda()).float()
    label_test = Variable(sample_batched['label'])#.cuda())
    output_test = the_model(data_test)
    loss = torch.sum(1. - output_test)

    correct = output_test.eq(label_test.float()).long().cpu().sum()



    print(output_test)

    print('Initial loss: ', float(loss), ', correct: ', int(correct),output_test)

    for i in range(1000000):
        input_data = torch.cat((equation.float(), answer.float()), 1)
        output = the_model(input_data)#.cuda())
        #loss = torch.sum(1. - output)
        loss = torch.sum(1.0 - output)/ batch_size
        if i % 1000 == 0:
            print('Iter: ', str(i), ', loss: ', float(loss))
            #print(output.shape)

        if i % 100 == 0:
            dump_solutions('img_solutions', equation, answer)
            print("dump", float(loss),output)
            with open("img_solutions\logs.csv",'a') as f:
                f.write(str(i)+","+",".join([str(x[0]) for x in output.detach().numpy()])+"\n")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < 0.00001:
            break
    break

# test_dataset = equation_linear_images_dataset_same_solution(100)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#
# i = 0
# success = 0
# total = 100
# for batch_idx, sample_batched in enumerate(test_loader):
#     i+=1
#     data = sample_batched['feature_vector'].numpy()[:,0,:,:].reshape((batch_size,1,30,100))
#     equation = Variable(torch.from_numpy(data), requires_grad=False)
#     input_data = torch.cat((equation.float(), answer.float()), 1)
#     output = the_model(input_data)
#     loss = torch.sum(1.0 - output)/ batch_size
#
#     print (output.detach().numpy()[0][0])
#     if output.detach().numpy()[0][0] > 0.5:
#         success += 1
#         dump_solutions('img_solutions_success', equation, answer,i)
#     else:
#         dump_solutions('img_solutions_fail', equation, answer,i)
#     if i >= total:
#         print (success / total * 100)
#         break
