from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import _pickle
import random

def eq_to_binary(a, x, b):
    def int_to_binary(i, num_digits):
        array_str = "{0:0"+str(num_digits)+"b}"
        return np.array(list(array_str.format(i)), dtype='float32')
    a_bin = int_to_binary(a, num_digits=10)
    x_bin = int_to_binary(x, num_digits=10)
    b_bin = int_to_binary(-b, num_digits=20)

    return np.concatenate((a_bin, x_bin, b_bin), axis=0)

class equation_binary_dataset_train(Dataset):

    def __init__(self, cv_set_file, right_asnwer_chance):
        self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
        self.right_answer_chance = right_asnwer_chance

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        is_in_cv_set = True
        while is_in_cv_set:
            a = np.random.randint(0, 1000)
            true_x = np.random.randint(0, 1000)
            is_in_cv_set = (a, true_x) in self.cv_set

        b = -a * true_x

        if np.random.random_sample() < self.right_answer_chance:
            x = true_x
            #label = np.array([1, 0], dtype=np.int64)
            label = 1
            weight = 1
        else:
            x = np.random.randint(0, 1000)
            while x == true_x: x = np.random.randint(0, 1000)
            #label = np.array([0, 1], dtype=np.int64)
            label = 0
            weight = abs(true_x - x)

        weight = np.array([weight, weight])

        sample = {'a': a, 'b': b, 'x': x, 'label': label, 'true_x': true_x, 'weight': weight, 'feature_vector': eq_to_binary(a, x, b)}

        return sample


class equation_binary_dataset_cv(Dataset):

    def __init__(self, cv_set_file, right_asnwer_chance):
        self.cv_set = _pickle.load(open(cv_set_file, 'rb'))
        self.right_answer_chance = right_asnwer_chance

    def __len__(self):
        return int(1.*len(self.cv_set)/self.right_answer_chance)

    def __getitem__(self, idx):
        sample = random.sample(self.cv_set, 1)[0]
        a = sample[0]
        true_x = sample[1]

        b = -a * true_x

        if np.random.random_sample() < self.right_answer_chance:
            x = true_x
            #label = np.array([1, 0], dtype=np.int64)
            label = 1
            weight = 1
        else:
            x = np.random.randint(0, 1000, dtype=np.int32)
            while x == true_x: x = np.random.randint(0, 1000, dtype=np.int32)
            #label = np.array([0, 1], dtype=np.int64)
            label = 0
            weight = abs(true_x - x)

        weight = np.array([weight, weight])

        sample = {'a': a, 'b': b, 'x': x, 'label': label, 'true_x': true_x, 'weight': weight, 'feature_vector': eq_to_binary(a, x, b)}

        return sample