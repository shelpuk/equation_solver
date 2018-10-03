from dataset_generator import *
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import stats

# from multiprocessing import freeze_support
#
# freeze_support()

# import matplotlib.pyplot as plt
# from dataset_generator_images import *
#
# #batch_size = 50
# def convert_to_image(img_array):
#     min = np.min(img_array)
#     max = np.max(img_array)
#     if min < 0:
#         img_array = 1.*(img_array - min) / ((1.*max / ((max - min)+0.0000001))+0.0000001)
#
#     img_array = img_array * 255
#     img_array = np.clip(img_array, a_min=0, a_max=255).astype(np.uint8)
#
#     img = Image.fromarray(img_array)
#
#     #img.show()
#
#     return img_array, img
#
# def dump_solutions(dir, equations, solutions,j):
#     equations_array = equations.data.numpy()
#     solutions_array = solutions.data.numpy()
#
#     for i in range(equations_array.shape[0]):
#         equation_array, _ = convert_to_image(equations_array[i, 0])
#         solution_array, _ = convert_to_image(solutions_array[i, 0])
#         stacked_image_array = np.concatenate((equation_array, solution_array), axis=0)
#         stacked_image = Image.fromarray(stacked_image_array)
#         #stacked_image.show()
#         stacked_image.save(dir+'/'+str(j)+'.png')
#
#
# path_to_folder = './eq_net_ff_07-31-16-27/'  # change here directory name
# the_model = torch.load(path_to_folder + 'model.pt', map_location='cpu')
# print(repr(the_model))
# weights = the_model.state_dict()
# #the_model = the_model.cuda()
# the_model.eval()
#
# # a = 914
# # #b= -127046
# # b = -507270
#
# a = 662
# b = -91356
#
# # a = 4
# # b = -3592
#
# # a = 530
# # b = -392730
#
# outputs = []
# for x in range(1024):
#     # images = generate_data_point(a, b,x)
#     #
#     # sample_eq = np.array(images[0]).reshape((1,1,30,100))
#     #
#     # #im = Image.fromarray(sample_eq.reshape((30,100)))
#     # #im.show()
#     #
#     # sample_answer = np.array(images[1]).reshape((1,1,30,100))
#     # equation = Variable(torch.from_numpy(sample_eq), requires_grad=False)
#     #
#     # # im = Image.fromarray(equation.numpy().reshape((30,100)))
#     # # im.show()
#     #
#     #
#     # answer = Variable(torch.from_numpy(sample_answer),requires_grad=False)
#     #
#     # input_data = torch.cat((equation.float(), answer.float()), 1)
#     # dump_solutions('img', equation, answer, x)
#
#     equation = eq_to_binary(a,x,b)
#     print(equation)
#     input_data = Variable(torch.from_numpy(equation.reshape((1,-1))), requires_grad=False)
#
#
#     output = the_model(input_data)
#     outputs.append(output.detach().numpy()[0,0])
#     print (x,output)
#     #dump_solutions('img',equation,answer,x)
#
# print (outputs)
#
# print(np.argmax(outputs))
#
# array_str = "{0:0"+str(10)+"b}"
#
# #n = [139,138,137,143,131,155,171,203,11,395,651]
# #n = [555,554,553,559,547,571,523,629,683,811,43]
# n = [138,139,136,142,130,154,170,202,10,394,650]
# # n = [682,683,680,686,674,698,650,746,554,938,170]
# y = []
# for i in n:
#     print(i,list(array_str.format(i)),outputs[i])
#     y.append(outputs[i]-outputs[n[0]])
#
# plt.plot(range(11),y)
# plt.show()
#
# plt.plot(range(1024),outputs)
# plt.show()


# def dump_solutions(dir, equations_a,equations_b, solutions,iteration):
#     equations_a_array = equations_a.data.numpy()
#     equations_b_array = equations_b.data.numpy()
#     solutions_array = solutions.data.numpy()
#
#     with open ("binary_solutions/solution_"+str(iteration)+".txt",'w') as f:
#         for i in range(equations_a_array.shape[0]):
#             f.write(str(equations_a_array[i]))
#             f.write(str(solutions_array[i]))
#             f.write(str(equations_b_array[i]))
# #         equation_array, _ = convert_to_image(equations_array[i, 0])
# #         solution_array, _ = convert_to_image(solutions_array[i, 0])
# #         stacked_image_array = np.concatenate((equation_array, solution_array), axis=0)
# #         stacked_image = Image.fromarray(stacked_image_array)
# #         #stacked_image.show()
# #         stacked_image.save(dir+'/'+str(i)+'.png')
#
# #test_dataset = equation_linear_images_dataset_cv(cv_set_file='cv_set_linear.p', right_asnwer_chance=1.0)


batch_size = 1
test_dataset = equation_binary_dataset_cv_4(cv_set_file='cv_set_4.p', right_answer_chance=1.0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

path_to_folder = './eq_net_ff_09-13-12-59/'  # change here directory name
the_model = torch.load(path_to_folder + 'model.pt')
print(repr(the_model))
weights = the_model.state_dict()
#the_model = the_model.cuda()
the_model.eval()

successful = 0
total = 0

# def dist_to_string(x):
#     d1 = x.abs()
#     d2 = (x - torch.tensor(np.array([1,1]).astype('float32'))).abs()
#
#     d_sum = d1 + d2
#     if (d1 + d2 - 1) < 0.000001:
#         d = 0
#     else:
#         d = min(d1,d2)
#     return d
#
# def distance(x):
#     d1 = dist_to_string(x)
#     d2 = dist_to_string(x)
#     distance = np.sqrt(d1 ** 2 + d2 ** 2)
#
#     return distance



for batch_idx, sample_batched in enumerate(test_loader):

    print(sample_batched['feature_vector'].numpy().shape)
    data = sample_batched['feature_vector'].numpy()#[:,0,:,:].reshape((batch_size,1,30,100))
    a = data[:,:10]
    count = batch_size- np.count_nonzero(np.all(a==0, axis=1))
    b = data[:,14:]
    correct_x = data[:,10:14]
    print(a.shape,b.shape)
    #print(data)

    equation_a = Variable(torch.from_numpy(a), requires_grad=False)
    equation_b = Variable(torch.from_numpy(b), requires_grad=False)

    #equation = Variable(torch.from_numpy(data), requires_grad=False)

    # Initialize answer as random noise
    solutions = []
    for i in range(10):
        answer = Variable(torch.from_numpy(np.random.normal(0.5,0.1,(batch_size,4)).astype('float32')), requires_grad=True)

    # Initialize answer as a 0
    #answer = Variable(torch.zeros(batch_size,10), requires_grad=True)

    #Initialize answer as a right answer
    #answer = Variable(torch.from_numpy(np.array(data[:,10:12]).astype('float32')), requires_grad=True)
        optimizer = optim.SGD([answer], lr=0.01, momentum=0.5)
    ##optimizer = optim.Adam([answer], lr=0.01)
    #optimizer = optim.RMSprop([answer], lr=0.001)




        optimizer.zero_grad()

        input_data = torch.cat((equation_a, answer,equation_b), 1)

        output = the_model(input_data)#.cuda())

        data_test = Variable(torch.from_numpy(sample_batched['feature_vector'].numpy())).float()#.cuda()).float()
        label_test = Variable(sample_batched['label'])#.cuda())
        output_test = the_model(data_test)
        print(output_test.detach().numpy().shape)
        loss = torch.sum(1. - output_test)

        correct = output_test.eq(label_test.float()).long().cpu().sum()



    #print(output_test)

        print('Initial loss: ', float(loss), ', correct: ', int(correct))

        for i in range(10000):
            input_data = torch.cat((equation_a, answer, equation_b), 1)
            output = the_model(input_data)#.cuda())
        # print(output.detach().numpy().shape)
        # loss = torch.sum(1. - output)
        # print (float(loss))
            x = answer.detach().numpy()[0]
            if np.all(x>=0) and  np.all(x<=1):
                loss = torch.sum(1.0 - output)/ batch_size
            else:
                loss = torch.sum(1.0 - output)/ batch_size + 20.0 * torch.sum(abs(answer-0.5)) - 9.0




            if i % 1000 == 0:
                print('Iter: ', str(i), ', loss: ', float(loss), output.detach().numpy().shape)
            #print(output.shape)

            #dump_solutions('img_solutions', equation_a,equation_b, answer,i)
            #print("dump", float(loss),output)
                result = answer.detach().numpy()
                result[result < 0.5] = 0.0
                result[result >= 0.5] = 1.0
                comparison = np.all(np.equal(result, correct_x), axis=1)

                acc = np.count_nonzero(comparison)

                print("Accuracy = ",acc)
                with open("img_solutions\logs.csv",'a') as f:
                    f.write(str(i)+","+",".join([str(x[0]) for x in output.detach().numpy()])+"\n")

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            x = answer.detach().numpy()[0]
            if not (np.all(x >= 0) and np.all(x <= 1)):
                answer.grad = 20.0 * (answer - torch.tensor(np.array([0.5,0.5,0.5,0.5]).astype('float32')))

            optimizer.step()

        # print ('gradient',answer.grad)

        # answer = torch.Tensor(torch.clamp(answer,min=0),requires_gradient=True)
        # answer = torch.Tensor(torch.clamp(answer,max=1),requires_gradient=True)

        #print (acc)

        #if loss < 0.00001:
         #   brea
        result = answer.detach().numpy()
        result[result < 0.5] = 0.0
        result[result >= 0.5] = 1.0
        comparison = np.all(np.equal(result, correct_x), axis=1)

        acc = np.count_nonzero(comparison)

        print("Accuracy = ", acc)

        #print (loss.detach().numpy())
        if loss.detach().numpy()< 0.5:
            solutions.append(result.reshape(4,))
        #print(result.shape)

    solutions = np.array(solutions)
    total += 1
    if solutions.shape[0] > 0:
        print(solutions)
    #print (stats.mode(solutions,axis = 0))
    #print (stats.mode(solutions,axis = 1))

        u, indices = np.unique(solutions, return_inverse=True, axis = 0)
    #print(u)
    #print(indices)
    #print(u[np.argmax(np.bincount(indices))])
        if np.all(np.equal(u[np.argmax(np.bincount(indices))], correct_x), axis=1)[0]:
            successful += 1
    print(successful, total)
    with open("get_solution_log_4.txt",'a') as f:
        f.write(str(successful)+','+str(total)+'\n')


print (successful, total)



#
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
