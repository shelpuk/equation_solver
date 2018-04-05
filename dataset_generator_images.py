from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import _pickle
import random
from PIL import Image, ImageDraw, ImageFont

def generate_images_linear(a, b, x):

    img_eq = Image.new('L', (100, 30), color=0)

    fnt = ImageFont.truetype('arial.ttf', 15)
    d = ImageDraw.Draw(img_eq)
    if b>=0: sign='+'
    else: sign = ''
    d.text((10, 10), str(a)+'x'+sign+str(b), font=fnt, fill=255)

    img_ans = Image.new('L', (100, 30), color=0)
    d = ImageDraw.Draw(img_ans)
    d.text((10, 10), str(x), font=fnt, fill=255)

    #img_eq.show()
    #img_ans.show()

    return img_eq, img_ans

#img_eq, img_ans = generate_images_linear(847,-424347,501)

#img_eq.show()
#img_ans.show()

def generate_data_point(a, b, x):
    img_eq, img_ans = generate_images_linear(a, b, x)
    img_eq_array = (np.array(img_eq) / 255)
    img_ans_array = np.array(img_ans) / 255

    return np.concatenate((img_eq_array.reshape(1, img_eq_array.shape[0], img_eq_array.shape[1]), img_ans_array.reshape(1, img_ans_array.shape[0], img_ans_array.shape[1])), axis=0)

class equation_linear_images_dataset_train(Dataset):

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
            label_array = np.array([0, 1])
            weight = 1
        else:
            x = np.random.randint(0, 1000)
            while x == true_x: x = np.random.randint(0, 1000)
            #label = np.array([0, 1], dtype=np.int64)
            label = 0
            label_array = np.array([1, 0])
            weight = abs(x - true_x)

        weight = np.array([np.log(weight+np.e-1)])
        #print(weight)
        label = np.array([label])

        sample = {'a': a, 'b': b, 'x': x, 'label': label, 'label_array': label_array, 'true_x': true_x, 'weight': weight, 'feature_vector': generate_data_point(a, b, x)}

        return sample


class equation_linear_images_dataset_cv(Dataset):

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
            label_array = np.array([0, 1])
            weight = 1
        else:
            x = np.random.randint(0, 1000, dtype=np.int32)
            while x == true_x: x = np.random.randint(0, 1000, dtype=np.int32)
            #label = np.array([0, 1], dtype=np.int64)
            label = 0
            label_array = np.array([1, 0])
            weight = abs(x - true_x)

        weight = np.array([np.log(weight+np.e-1)])
        label = np.array([label])


        sample = {'a': a, 'b': b, 'x': x, 'label': label, 'label_array': label_array, 'true_x': true_x, 'weight': weight, 'feature_vector': generate_data_point(a, b, x)}

        return sample


