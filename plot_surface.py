import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from dataset_generator import *
from torch.autograd import Variable


a = int_to_binary(285,10)
b = int_to_binary(285*2,20)

path_to_folder = './eq_net_ff_08-29-18-35/'  # change here directory name
the_model = torch.load(path_to_folder + 'model.pt')
print(repr(the_model))
weights = the_model.state_dict()
#the_model = the_model.cuda()
the_model.eval()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

x = np.linspace(0.0,1.0)
y = np.linspace(0.0,1.0)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T

Z = np.zeros((X.shape[0],Y.shape[0]))

def dist_to_string(x,i):
    d1 = abs(x[i])
    d2 = abs(x[i] - 1)

    if (d1 + d2 - 1) < 0.000001:
        d = 0
    else:
        d = min(d1,d2)
    return d

def distance(x):
    d1 = dist_to_string(x,0)
    d2 = dist_to_string(x,1)
    distance = np.sqrt(d1 ** 2 + d2 ** 2)

    return distance

print(distance([1,-2]))


for i in  range(Z.shape[0]):
    for j in range(Z.shape[1]):
        #print(a,X[i,j],Y[i,j],b)
        equation = np.concatenate((a,[X[i,j]],[Y[i,j]],b)).astype('float32')
        input_data = Variable(torch.from_numpy(equation.reshape((1, -1))), requires_grad=False)
        output = the_model(input_data)
        input_x = np.array([X[i,j],Y[i,j]])
        print(input_x)
        if np.all(input_x>=0) and  np.all(input_x<=1):
            loss = torch.sum(1.0 - output)
        else:
            loss = 20.0 * distance(input_x) + 1.0
        #print(output)
        Z[i,j] =loss


from matplotlib import cm
#print X.shape, Y.shape, Z.shape
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
plt.show()