from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import _pickle
import random


def int_to_binary(i, num_digits):
    array_str = "{0:0" + str(num_digits) + "b}"
    return np.array(list(array_str.format(i)), dtype='float32')

def binary_to_int(i,binary):
    #print("binary_to_int",binary)
    s = binary[i-1]
    m = 2
    for j in range(i-2,-1,-1):
        s+= m*binary[j]
        m *= 2
    return int(s)

def eq_to_binary(a, x, b):
    #print (a,x,b)
    a_bin = int_to_binary(a, num_digits=10)
    x_bin = int_to_binary(x, num_digits=10)
    b_bin = int_to_binary(-b, num_digits=20)

    return np.concatenate((a_bin, x_bin, b_bin), axis=0)

def eq_to_binary_2(a, x_bin, b):
    #print (a,x,b)
    a_bin = int_to_binary(a, num_digits=10)
    #x_bin = int_to_binary(x, num_digits=2)
    b_bin = int_to_binary(-b, num_digits=20)
    #print(np.concatenate((a_bin, x_bin, b_bin), axis=0))
    return np.concatenate((a_bin, x_bin, b_bin), axis=0)

class equation_binary_dataset(Dataset):
    def __init__(self, cv_set_file,right_answer_chance, wrong_answer_generator,weight_function,n,is_train = True, ):
        self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
        self.right_answer_chance = right_answer_chance
        self.wrong_answer_generator = wrong_answer_generator
        self.is_train = is_train
        self.weight_function = weight_function
        self.n = n

    def __len__(self):
        if not self.is_train:
            return int(1. * (len(self.cv_set)//50)*50 / self.right_answer_chance)
        else:
            return 10000

    def __getitem__(self, idx):
        if self.is_train:
            is_in_cv_set = True
            while is_in_cv_set:
                a = np.random.randint(0, 1024)
                true_x = np.random.randint(0, 2**self.n)
                is_in_cv_set = (a, true_x) in self.cv_set
        else:
            sample = random.sample(self.cv_set, 1)[0]
            a = sample[0]
            true_x = sample[1]

        b = -a * true_x

        if np.random.random_sample() < self.right_answer_chance:
            binary_x = int_to_binary(true_x,self.n)
            label = 1
            label_array = np.array([0, 1])
            weight = 1

        else:
            binary_x = self.wrong_answer_generator(true_x,self.n)
            #print(binary_x.shape)
            label = 0
            label_array = np.array([1, 0])
            weight = self.weight_function(binary_x,int_to_binary(true_x,self.n))
            if a == 0:
                label = 1
                label_array = np.array([0, 1])
                weight = 1
        label = np.array([label])

        sample = {'a': a, 'b': b, 'x': binary_x, 'label': label, 'label_array': label_array, 'true_x': true_x,
                        'weight': weight, 'feature_vector': eq_to_binary_2(a, binary_x, b)}

        #print (sample)

        return sample

def generate_wrong_x_continuous(true_x):
    binary_true_x = int_to_binary(true_x,2)
    binary_x = np.copy(binary_true_x)
    while np.all(binary_true_x==binary_x):
        for i in range(2):
            if np.random.random_sample() < 0.5:
                binary_x[i] = np.random.random()
    return binary_x

def generate_wrong_x(true_x,n):
    x = np.random.randint(0, 2**n, dtype=np.int32)
    while x == true_x: x = np.random.randint(0, 2**n, dtype=np.int32)
    return int_to_binary(x,n)

def generate_wrong_x_flip(true_x,n):
    binary_true_x = int_to_binary(true_x,n)
    binary_x = np.copy(binary_true_x)
    while np.all(binary_true_x==binary_x):
        for i in range(n):
            if np.random.random_sample() < 1.0/n:
                if binary_true_x[i] == 0:
                    binary_x[i] = 1
                else:
                    binary_x[i] = 0
    return binary_x


def manhattan_distance(binary_x, binary_true_x):
    #print (np.sum(np.abs(binary_x - binary_true_x)))
    return np.sum(np.abs(binary_x - binary_true_x))


equation_binary_dataset_cv_2 = lambda cv_set_file,right_answer_chance : equation_binary_dataset(cv_set_file=cv_set_file, right_answer_chance=right_answer_chance,
                                                                                                wrong_answer_generator=generate_wrong_x,
                                                                                                weight_function=manhattan_distance,n=2,is_train=False)

equation_binary_dataset_train_2 = lambda cv_set_file,right_answer_chance : equation_binary_dataset(cv_set_file=cv_set_file, right_answer_chance=right_answer_chance,
                                                                                                wrong_answer_generator= generate_wrong_x_flip,
                                                                                                   weight_function= manhattan_distance,n=2,is_train=True)

equation_binary_dataset_cv_3 = lambda cv_set_file,right_answer_chance : equation_binary_dataset(cv_set_file=cv_set_file, right_answer_chance=right_answer_chance,
                                                                                                wrong_answer_generator=generate_wrong_x,
                                                                                                weight_function=manhattan_distance,n=3,is_train=False)

equation_binary_dataset_train_3 = lambda cv_set_file,right_answer_chance : equation_binary_dataset(cv_set_file=cv_set_file, right_answer_chance=right_answer_chance,
                                                                                                wrong_answer_generator= generate_wrong_x_flip,
                                                                                                   weight_function= manhattan_distance,n=3,is_train=True)

equation_binary_dataset_cv_4 = lambda cv_set_file,right_answer_chance : equation_binary_dataset(cv_set_file=cv_set_file, right_answer_chance=right_answer_chance,
                                                                                                wrong_answer_generator=generate_wrong_x,
                                                                                                weight_function=manhattan_distance,n=4,is_train=False)

equation_binary_dataset_train_4 = lambda cv_set_file,right_answer_chance : equation_binary_dataset(cv_set_file=cv_set_file, right_answer_chance=right_answer_chance,
                                                                                                wrong_answer_generator= generate_wrong_x,
                                                                                                   weight_function= manhattan_distance,n=4,is_train=True)


equation_binary_dataset_cv = lambda cv_set_file,right_answer_chance : equation_binary_dataset(cv_set_file=cv_set_file, right_answer_chance=right_answer_chance,
                                                                                                wrong_answer_generator=generate_wrong_x,
                                                                                                weight_function=manhattan_distance,n=10,is_train=False)

equation_binary_dataset_train = lambda cv_set_file,right_answer_chance : equation_binary_dataset(cv_set_file=cv_set_file, right_answer_chance=right_answer_chance,
                                                                                                wrong_answer_generator= generate_wrong_x_flip,
                                                                                                   weight_function= manhattan_distance,n=10,is_train=True)

# class equation_binary_dataset_cv_2(Dataset):
#     def __init__(self, cv_set_file, right_answer_chance):
#         self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
#         self.right_answer_chance = right_answer_chance
#
#
#     def __len__(self):
#         return int(1.*len(self.cv_set)/self.right_answer_chance)
#
#     def __getitem__(self, idx):
#         sample = random.sample(self.cv_set, 1)[0]
#         a = sample[0]
#         true_x = sample[1]
#
#         b = -a * true_x
#
#         if np.random.random_sample() < self.right_answer_chance:
#             x = true_x
#              #label = np.array([1, 0], dtype=np.int64)
#             label = 1
#             label_array = np.array([0, 1])
#             weight = 1
#
#         else:
#             x = np.random.randint(0, 4, dtype=np.int32)
#             while x == true_x: x = np.random.randint(0, 4, dtype=np.int32)
#             #label = np.array([0, 1], dtype=np.int64)
#             label = 0
#             if a == 0:
#                 label = 1
#             label_array = np.array([1, 0])
#                   #weight = np.linalg.norm(int_to_binary(x,2)- int_to_binary(true_x,2))#np.count_nonzero(int_to_binary(x,2)!=int_to_binary(true_x,2))
#             weight = np.sum(np.abs(int_to_binary(x, 2) - int_to_binary(true_x, 2)))
#                #weight = 1#np.array([np.log(weight+np.e-1)])
#         label = np.array([label])
#
#
#         sample = {'a': a, 'b': b, 'x': x, 'label': label, 'label_array': label_array, 'true_x': true_x, 'weight': weight, 'feature_vector': eq_to_binary_2(a, int_to_binary(x,2), b)}
#         print(sample)
#         return sample
#
# class equation_binary_dataset_train_flip_2(Dataset):
#
#     def __init__(self, cv_set_file, right_answer_chance):
#         self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
#         self.right_answer_chance = right_answer_chance
#
#     def __len__(self):
#         return 10000
#
#     def generate_sample_with_flip(self,true_x):
#         binary_true_x = int_to_binary(true_x,2)
#         binary_x = np.copy(binary_true_x)
#                #print("generate", true_x)
#         while np.all(binary_true_x==binary_x):
#                     #print("in loop")
#             for i in range(2):
#                 if np.random.random_sample() < 0.5:
#                             # if binary_true_x[i] == 0:
#                             #     #print ("binary_x",binary_x,binary_true_x)
#                             #     binary_x[i]=1
#                             # else:
#                             #     binary_x[i] = 0
#                     binary_x[i] = np.random.random()
#                             #print(binary_x[i])
#                 #print(binary_x)
#             return binary_x
#
#     def __getitem__(self, idx):
#         is_in_cv_set = True
#         while is_in_cv_set:
#             a = np.random.randint(0, 1024)
#             true_x = np.random.randint(0, 4)
#             is_in_cv_set = (a, true_x) in self.cv_set
#
#             b = -a * true_x
#
#         if np.random.random_sample() < self.right_answer_chance:
#             x = true_x
#             binary_x = int_to_binary(x,2)
#             #label = np.array([1, 0], dtype=np.int64)
#             label = 1
#             label_array = np.array([0, 1])
#             weight = 1
#         else:
#             n = int(np.random.random_sample()/0.1)
#             binary_x = self.generate_sample_with_flip(true_x)
#             label = 0
#             if a == 0:
#                 label=1
#             label_array = np.array([1, 0])
#                     #weight = np.linalg.norm(int_to_binary(x,2)- int_to_binary(true_x,2))#np.count_nonzero(int_to_binary(x,2)!=int_to_binary(true_x,2))
#             weight = np.sum(np.abs(binary_x - int_to_binary(true_x, 2)))
#
#                 #weight = 1#np.array([np.log(weight+np.e-1)])
#                 #print(weight)
#         label = np.array([label])
#
#         sample = {'a': a, 'b': b, 'x': binary_x, 'label': label, 'label_array': label_array, 'true_x': true_x, 'weight': weight, 'feature_vector': eq_to_binary_2(a, binary_x, b)}
#
#         return sample


# class equation_binary_dataset_train(Dataset):
#
#     def __init__(self, cv_set_file, right_asnwer_chance):
#         self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
#         self.right_answer_chance = right_asnwer_chance
#
#     def __len__(self):
#         return 1000000
#
#     def __getitem__(self, idx):
#         is_in_cv_set = True
#         while is_in_cv_set:
#             a = np.random.randint(0, 1000)
#             true_x = np.random.randint(0, 1000)
#             is_in_cv_set = (a, true_x) in self.cv_set
#
#         b = -a * true_x
#
#         if np.random.random_sample() < self.right_answer_chance:
#             x = true_x
#             #label = np.array([1, 0], dtype=np.int64)
#             label = 1
#             label_array = np.array([0, 1])
#             weight = 1
#         else:
#             x = np.random.randint(0, 1000)
#             #x = np.random.randint(max(true_x-1,0),min(true_x+1,1000))
#             while x == true_x: x = np.random.randint(0, 1000)
#             if np.random.random_sample()< 0.25:
#                 x = max(true_x - np.random.randint(1, 50),0)
#             elif np.random.random_sample()< 0.5:
#                 x = min(true_x + np.random.randint(1, 50),999)
#
#             #label = np.array([0, 1], dtype=np.int64)
#             label = 0
#             if a == 0:
#                 label = 1
#             label_array = np.array([1, 0])
#             weight = 1#abs(x - true_x)
#
#         weight = 1#np.array([np.log(weight+np.e-1)])
#         #print(weight)
#         label = np.array([label])
#
#         sample = {'a': a, 'b': b, 'x': x, 'label': label, 'label_array': label_array, 'true_x': true_x, 'weight': weight, 'feature_vector': eq_to_binary(a, x, b)}
#
#         return sample
#
#
# class equation_binary_dataset_cv(Dataset):
#
#     def __init__(self, cv_set_file, right_asnwer_chance):
#         self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
#         self.right_answer_chance = right_asnwer_chance
#
#
#     def __len__(self):
#         return int(1.*len(self.cv_set)/self.right_answer_chance)
#
#     def __getitem__(self, idx):
#         sample = random.sample(self.cv_set, 1)[0]
#         a = sample[0]
#         true_x = sample[1]
#
#         b = -a * true_x
#
#         if np.random.random_sample() < self.right_answer_chance:
#             x = true_x
#             #label = np.array([1, 0], dtype=np.int64)
#             label = 1
#             label_array = np.array([0, 1])
#             weight = 1
#
#         else:
#             x = np.random.randint(0, 1000, dtype=np.int32)
#             while x == true_x: x = np.random.randint(0, 1000, dtype=np.int32)
#             #label = np.array([0, 1], dtype=np.int64)
#             label = 0
#             if a == 0:
#                 label = 1
#             label_array = np.array([1, 0])
#             weight = np.count_nonzero(int_to_binary(x,10)!=int_to_binary(true_x,10))
#
#         #weight = 1#np.array([np.log(weight+np.e-1)])
#         label = np.array([label])
#
#
#         sample = {'a': a, 'b': b, 'x': x, 'label': label, 'label_array': label_array, 'true_x': true_x, 'weight': weight, 'feature_vector': eq_to_binary(a, x, b)}
#
#         return sample
#
# class equation_binary_dataset_train_prob(Dataset):
#
#     def __init__(self, cv_set_file, right_asnwer_chance):
#         self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
#         self.right_answer_chance = right_asnwer_chance
#
#     def __len__(self):
#         return 1000000
#
#     def generate_sample_with_fixed_tail(self,true_x,n):
#         x = true_x
#         #print("generate", n)
#         while x == true_x:
#             #print("in loop")
#             x = np.random.randint(0, (1024 // (2 ** n))) + true_x % (2 ** n)
#         return x
#
#     def __getitem__(self, idx):
#         is_in_cv_set = True
#         while is_in_cv_set:
#             a = np.random.randint(0, 1024)
#             true_x = np.random.randint(0, 1024)
#             is_in_cv_set = (a, true_x) in self.cv_set
#
#         b = -a * true_x
#
#         if np.random.random_sample() < self.right_answer_chance:
#             x = true_x
#             #label = np.array([1, 0], dtype=np.int64)
#             label = 1
#             label_array = np.array([0, 1])
#             weight = 1
#         else:
#             n = int(np.random.random_sample()/0.1)
#             x = self.generate_sample_with_fixed_tail(true_x,n)
#             label = 0
#             if a == 0:
#                 label=1
#             label_array = np.array([1, 0])
#             weight = 1#abs(x - true_x)
#
#         weight = 1#np.array([np.log(weight+np.e-1)])
#         #print(weight)
#         label = np.array([label])
#
#         sample = {'a': a, 'b': b, 'x': x, 'label': label, 'label_array': label_array, 'true_x': true_x, 'weight': weight, 'feature_vector': eq_to_binary(a, x, b)}
#
#         return sample
#
# class equation_binary_dataset_cv_prob(Dataset):
#
#     def __init__(self, cv_set_file, right_asnwer_chance):
#         self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
#         self.right_answer_chance = right_asnwer_chance
#
#     def generate_sample_with_fixed_tail(self, true_x, n):
#         x = true_x
#         # print("generate", n)
#         while x == true_x:
#             # print("in loop")
#             x = np.random.randint(0, (1024 // (2 ** n))) + true_x % (2 ** n)
#         return x
#
#
#     def __len__(self):
#         return int(1.*len(self.cv_set)/self.right_answer_chance)
#
#     def __getitem__(self, idx):
#         sample = random.sample(self.cv_set, 1)[0]
#         a = sample[0]
#         true_x = sample[1]
#
#         b = -a * true_x
#
#         if np.random.random_sample() < self.right_answer_chance:
#             x = true_x
#             #label = np.array([1, 0], dtype=np.int64)
#             label = 1
#             label_array = np.array([0, 1])
#             weight = 1
#
#         else:
#             n = int(np.random.random_sample() / 0.1)
#             x = self.generate_sample_with_fixed_tail(true_x, n)
#             label = 0
#             if a == 0:
#                 label = 1
#             label_array = np.array([1, 0])
#             weight = 1#abs(x - true_x)
#
#         weight = 1#np.array([np.log(weight+np.e-1)])
#         label = np.array([label])
#
#
#         sample = {'a': a, 'b': b, 'x': x, 'label': label, 'label_array': label_array, 'true_x': true_x, 'weight': weight, 'feature_vector': eq_to_binary(a, x, b)}
#
#         return sample
#
# class equation_binary_dataset_train_flip(Dataset):
#
#     def __init__(self, cv_set_file, right_asnwer_chance):
#         self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
#         self.right_answer_chance = right_asnwer_chance
#
#     def __len__(self):
#         return 1000000
#
#     def generate_sample_with_flip(self,true_x):
#         binary_true_x = int_to_binary(true_x,10)
#         binary_x = np.copy(binary_true_x)
#         #print("generate", true_x)
#         while np.all(binary_true_x==binary_x):
#             #print("in loop")
#             for i in range(10):
#                 if np.random.random_sample() < 0.1:
#                     if binary_true_x[i] == 0:
#                         #print ("binary_x",binary_x,binary_true_x)
#                         binary_x[i]=1
#                     else:
#                         binary_x[i] = 0
#         #print(binary_x)
#         return binary_to_int(10,binary_x)
#
#     def __getitem__(self, idx):
#         is_in_cv_set = True
#         while is_in_cv_set:
#             a = np.random.randint(0, 1024)
#             true_x = np.random.randint(0, 1024)
#             is_in_cv_set = (a, true_x) in self.cv_set
#
#         b = -a * true_x
#
#         if np.random.random_sample() < self.right_answer_chance:
#             x = true_x
#             #label = np.array([1, 0], dtype=np.int64)
#             label = 1
#             label_array = np.array([0, 1])
#             weight = 1
#         else:
#             n = int(np.random.random_sample()/0.1)
#             x = self.generate_sample_with_flip(true_x)
#             label = 0
#             if a == 0:
#                 label=1
#             label_array = np.array([1, 0])
#             weight = np.count_nonzero(int_to_binary(x,10)!=int_to_binary(true_x,10))
#
#         #weight = 1#np.array([np.log(weight+np.e-1)])
#         #print(weight)
#         label = np.array([label])
#
#         sample = {'a': a, 'b': b, 'x': x, 'label': label, 'label_array': label_array, 'true_x': true_x, 'weight': weight, 'feature_vector': eq_to_binary(a, x, b)}
#
#         return sample
#
# class equation_binary_dataset_train_flip_n(Dataset):
#
#     def __init__(self, cv_set_file, right_asnwer_chance,n):
#         self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
#         self.right_answer_chance = right_asnwer_chance
#         self.n = n
#
#     def __len__(self):
#         return 1000000
#
#     def generate_sample_with_flip_n(self,true_x):
#         binary_true_x = int_to_binary(true_x,10)
#         binary_x = np.copy(binary_true_x)
#         #print("generate", true_x)
#         while np.all(binary_true_x==binary_x):
#             #print("in loop")
#             flip_indices = np.random.choice(10,np.random.randint(0,self.n+1))
#             for i in flip_indices:
#                 if binary_true_x[i] == 0:
#                         #print ("binary_x",binary_x,binary_true_x)
#                     binary_x[i]=1
#                 else:
#                     binary_x[i] = 0
#         #print(binary_x)
#         return binary_to_int(10,binary_x)
#
#     def __getitem__(self, idx):
#         is_in_cv_set = True
#         while is_in_cv_set:
#             a = np.random.randint(0, 1024)
#             true_x = np.random.randint(0, 1024)
#             is_in_cv_set = (a, true_x) in self.cv_set
#
#         b = -a * true_x
#
#         if np.random.random_sample() < self.right_answer_chance:
#             x = true_x
#             #label = np.array([1, 0], dtype=np.int64)
#             label = 1
#             label_array = np.array([0, 1])
#             weight = 1
#         else:
#             n = int(np.random.random_sample()/0.1)
#             x = self.generate_sample_with_flip_n(true_x)
#             label = 0
#             if a == 0:
#                 label=1
#             label_array = np.array([1, 0])
#             weight = 1#abs(x - true_x)
#
#         weight = 1#np.array([np.log(weight+np.e-1)])
#         #print(weight)
#         label = np.array([label])
#
#         sample = {'a': a, 'b': b, 'x': x, 'label': label, 'label_array': label_array, 'true_x': true_x, 'weight': weight, 'feature_vector': eq_to_binary(a, x, b)}
#
#         return sample
#
# class equation_binary_dataset_train_flip_min_n(Dataset):
#
#     def __init__(self, cv_set_file, right_asnwer_chance,n):
#         self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
#         self.right_answer_chance = right_asnwer_chance
#         self.n = n
#
#     def __len__(self):
#         return 1000000
#
#     def generate_sample_with_flip_n(self,true_x):
#         binary_true_x = int_to_binary(true_x,10)
#         binary_x = np.copy(binary_true_x)
#         #print("generate", true_x)
#         while np.all(binary_true_x==binary_x):
#             #print("in loop")
#             flip_indices = np.random.choice(10,np.random.randint(self.n,11))
#             for i in flip_indices:
#                 if binary_true_x[i] == 0:
#                         #print ("binary_x",binary_x,binary_true_x)
#                     binary_x[i]=1
#                 else:
#                     binary_x[i] = 0
#         #print(binary_x)
#         return binary_to_int(10,binary_x)
#
#     def __getitem__(self, idx):
#         is_in_cv_set = True
#         while is_in_cv_set:
#             a = np.random.randint(0, 1024)
#             true_x = np.random.randint(0, 1024)
#             is_in_cv_set = (a, true_x) in self.cv_set
#
#         b = -a * true_x
#
#         if np.random.random_sample() < self.right_answer_chance:
#             x = true_x
#             #label = np.array([1, 0], dtype=np.int64)
#             label = 1
#             label_array = np.array([0, 1])
#             weight = 1
#         else:
#             n = int(np.random.random_sample()/0.1)
#             x = self.generate_sample_with_flip_n(true_x)
#             label = 0
#             if a == 0:
#                 label=1
#             label_array = np.array([1, 0])
#             weight = 1#abs(x - true_x)
#
#         weight = 1#np.array([np.log(weight+np.e-1)])
#         #print(weight)
#         label = np.array([label])
#
#         sample = {'a': a, 'b': b, 'x': x, 'label': label, 'label_array': label_array, 'true_x': true_x, 'weight': weight, 'feature_vector': eq_to_binary(a, x, b)}
#
#         return sample
#
