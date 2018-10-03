import numpy as np
import matplotlib.pyplot as plt
import _pickle

# with open ("img_solutions\logs.csv") as f:
#     data_parsed = []
#     for line in f.readlines():
#         data_parsed.append(line.split(","))
#
# log = np.array(data_parsed)
# print(log.shape)
#
# for i in range(50):
#     plt.title(str(i))
#     plt.plot(log[:,0], log[:,i+1])
#     plt.show()

cv_set = _pickle.load(open('cv_set_linear.p', 'rb'))
print (cv_set)