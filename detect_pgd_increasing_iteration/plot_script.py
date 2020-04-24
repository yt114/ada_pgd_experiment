import matplotlib.pyplot as plt
import numpy as np

import re

# def read_auc(lines):
#     it = []
#     auc = []
#     for l in lines:
#         data = l.strip('\n')
#         data = re.split(r'[:,\s]\s*', data)
#         it.append(int(data[1]))
#         auc.append(float(data[3]))
#
#         print(l)
#     return it, auc
#
# with open('./MNIST_full_kl_detection.txt', 'r') as f:
#     ls = f.readlines()
#
#
# it1, auc1 = read_auc(ls)
# print(it1)
# print(auc1)
# with open('./MNIST_target_class_detection.txt', 'r') as f:
#     ls = f.readlines()
#
# it2, auc2 = read_auc(ls)
#
# plt.figure()
#
# plt.plot(it1, auc1, '-*', label='full kl method')
# plt.plot(it2, auc2, '-*', label='target class density method')
# plt.legend()
# plt.ylim([0.5, 1])
# plt.ylabel('AUC-ROC')
# plt.xlabel('Iteration')
# plt.grid()
# plt.show()
#

def read_attack(fn):
    with open(fn, 'r') as f:
        ls = f.readlines()

    res = []
    for l in ls:
        l = l.strip('\n ')
        print(l)
        data = re.split(r'[:,\s]\s', l)
        temp = [data[1], data[9], data[11]]
        res.append(' & '.join(temp) +'\\\\\n')

    return res

header = ['Iteration', 'l2 norm distance', 'Confidence']
hl = ' & '.join(header) + '\\\\ \n'
latex_lines = read_attack('./PGD_attack_data.txt')
latex_lines = [hl] + latex_lines
print(latex_lines)

with open('latex_pgd_attack_mnist.txt', 'w+') as f:
    f.writelines(latex_lines)
