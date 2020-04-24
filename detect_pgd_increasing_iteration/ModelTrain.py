from modelZoo import LeNet5, ResNet18, ResNet16, MnistNet, PytorchResNet18
from NNmodels import TrainNN
import torch
import torchvision
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import argparse
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

ATTACK_PARAMS = {
    'mnist': {'eps': 0.300, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010}
}


def generate_attack_targets(y_c, class_num=10):
    ''' Randomly choose targets other than the correct lable
    '''
    attack_targets =torch.randint(0, class_num, y_c.shape)
    for i in range(attack_targets.shape[0]):
        if attack_targets[i] == y_c[i]:
            attack_targets[i] = (attack_targets[i] +1) % (class_num)

    return attack_targets

def attacking(data_test, modelTrain,attacker, bz, save_fn, is_targeted):
    # perform attack here on test
    success_attack_ims = None
    success_attack_labels = None

    data_test_loader = DataLoader(data_test, bz , num_workers=8)
    success_attack = 0
    total_ims = 0

    l2norm = 0
    i=0
    for x, y in data_test_loader:
        ind_wrong, ind_correct = modelTrain.test_one_batch(x, y, verbose=0)

        if ind_correct.shape[0] == 0:
            continue
        ind_correct = ind_correct.squeeze(dim=1)

        x_correct = x[ind_correct]
        y_correct = y[ind_correct]

        x_correct = x_correct.to(modelTrain.dev)
        y_correct = y_correct.to(modelTrain.dev)
        if is_targeted:
            y_attack_targets = generate_attack_targets(y_correct)
            # x_adv =attacker.forward(x_correct, y_attack_targets)
            # x_adv = attacker(modelTrain.net, x_correct, y_attack_targets, to_numpy=False)
            y_attack_targets = y_attack_targets.to(modelTrain.dev)

            x_adv = attacker.perturb(x_correct, y_attack_targets)
            #ind_wrong, _ = modelTrain.test_one_batch(x_adv, y_correct, verbose=0)
            print(y_attack_targets)
        else:
            x_adv =attacker.perturb(x_correct, y_correct)

        x_adv.to(modelTrain.dev)
        ind_wrong, _ = modelTrain.test_one_batch(x_adv, y_correct, verbose=1)
        ind_wrong = ind_wrong.squeeze(dim=1)
        success_attack += ind_wrong.shape[0]
        total_ims += ind_correct.shape[0]

        l2norm += (torch.norm(x_adv[ind_wrong].to(modelTrain.dev)-x_correct[ind_wrong].to(modelTrain.dev), 2).sum().item())

        if success_attack_ims is not None:
            success_attack_ims = torch.cat([success_attack_ims, x_adv[ind_wrong]], dim=0)
            success_attack_labels = torch.cat([success_attack_labels, y_correct[ind_wrong]], dim=0)
        else:
            success_attack_ims = x_adv[ind_wrong]
            success_attack_labels = y_correct[ind_wrong]

        if i > 10:
            print(success_attack, total_ims)
            print('l2 norm', l2norm/success_attack)
            print('success attack ratio: ', success_attack/total_ims)

        # im_normal = np.squeeze(x_correct[ind_wrong], axis=0).cpu().clone().detach()
        # im_adv = np.squeeze(x_adv[ind_wrong], axis=0).cpu().clone().detach()
        # img = torchvision.utils.make_grid([im_normal, im_adv])
        #
        # plt.figure()
        # npimg = img.numpy()
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()
        #
        # plt.close()

        # if i == 2:
        #     break

        i+= 1
        print(i)

    advs = {'image': success_attack_ims, 'label': success_attack_labels}
    torch.save(advs, save_fn)
    print(success_attack, total_ims, l2norm/success_attack)



def main():
    if args.model == 'MNIST':
        data_train = MNIST('./data/mnist', download=True,
                           transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

        data_test = MNIST('./data/mnist', train=False, download=True,
                          transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
        data_train_loader = DataLoader(data_train, batch_size=256, shuffle=False, num_workers=8)
        data_test_loader = DataLoader(data_test, batch_size=256, num_workers=8, shuffle=False)
        model = LeNet5()
        prefix = 'MnistNet'
        modelTrain = TrainNN(prefix, net=model)
        if args.resume is None:
            modelTrain.train(51, data_train_loader, data_test_loader, writer, cpk=[20, 30, 50])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet_v1', required=False)
    parser.add_argument('--resume', type=str, default=None, required=False)

    args = parser.parse_args()
    print(vars(args))
    main()