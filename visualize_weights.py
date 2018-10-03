import torch
import matplotlib.pyplot as plt

path_to_folder = './eq_net_ff_dropout_07-26-13-32/'  # change here directory name
the_model = torch.load(path_to_folder + 'model.pt', map_location='cpu')
print(repr(the_model))
weights = the_model.state_dict()
#the_model = the_model.cuda()
the_model.eval()

i = 0
for m in the_model.modules():
    if isinstance(m, torch.nn.Linear):
        w = m.weight.data.numpy()
        plt.imshow(w, cmap="YlGnBu", interpolation='nearest')
        plt.show()
        i+=1
        if i == 11:
            plt.plot(range(400),w.flatten())
            plt.show()