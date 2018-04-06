from dataset_generator_images import *
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

test_dataset = equation_linear_images_dataset_cv(cv_set_file='cv_set_linear.p', right_asnwer_chance=.5)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=16)

path_to_folder = './LeNet_04-06-20:52/'  # change here directory name
the_model = torch.load(path_to_folder + 'model.pt')
print(repr(the_model))
weights = the_model.state_dict()

correct = []
batch_size = 50

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
    data, label, label_array, weight = Variable(data), Variable(label), Variable(label_array), Variable(
        weight).float()

    output = the_model(data.float())
    pred = torch.round(output)
    correct_ = sum(pred.long() == label).cpu().data.numpy()[0] / batch_size
    correct.append(correct_)
    # correct += pred.eq(label.float()).long().cpu().sum()

print("Average accuracy: {}".format(np.mean(correct)))
