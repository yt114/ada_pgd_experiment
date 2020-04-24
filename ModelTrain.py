from modelZoo import LeNet5, ResNet18, ResNet16, MnistNet, PytorchResNet18
from NNmodels import TrainNN
import torch
import torchvision
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from CWattack import CW
from cw import L2Adversary as CW
writer = SummaryWriter()
import argparse
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from pgdAttack import PGDattack
from collections import OrderedDict
from Resnet_v2 import resnet_18_v2

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

def get_robust_classifer_sdic(resumepath):
    # original saved file with DataParallel
    state_dict = torch.load(args.resume)
    # create new OrderedDict that does not contain `module.`

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        if name == "linear.weight":
            name = "linear1.weight"
        if name == 'linear.bias':
            name = 'linear1.bias'
        new_state_dict[name] = v

    return new_state_dict


def main():
    if args.model == 'MNIST':
        data_train = MNIST('./data/mnist', download=True,
                           transform=transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()]))

        data_test = MNIST('./data/mnist', train=False, download=True,
                          transform=transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()]))
        data_train_loader = DataLoader(data_train, batch_size=256, shuffle=False, num_workers=8)
        data_test_loader = DataLoader(data_test, batch_size=256, num_workers=8, shuffle=False)
        # model = LeNet5()
        # prefix = 'LeNet5-mnist-dropout'

        model = MnistNet()
        prefix = 'MnistNet'
        modelTrain = TrainNN(prefix, net=model)
        if args.resume is None:
            modelTrain.train(20, data_train_loader, data_test_loader, writer)
            modelTrain.save_checkpoint(20)
        else:
            modelTrain.load_checkpoint(args.resume)
            modelTrain.test(20, data_train_loader)
            modelTrain.test(20, data_test_loader)
            ## trian untargeted attack
            # cwAttacker = CW(modelTrain.net
            #                 , modelTrain.dev, targeted=False, c=13, kappa=4, iters=6000, lr=0.01)
            # cw_attack(data_test, modelTrain, cwAttacker, 128, './attackIms/mnist_adv.pth')

            # train targeted attack

            # cw l2 targeted attack
            # cwAttacker = CW(modelTrain.net
            #                 , modelTrain.dev, targeted=True, c=4, kappa=3, iters=200, lr=0.1)
            # attacking(data_test, modelTrain, cwAttacker, 2, './attackIms/MnistNet_cw_targeted.pth', is_targeted=True)

            Attacker = CW(modelTrain.net, targeted=True, c_range=(3, 3.0001),
                          confidence=1,
                          search_steps=1, box=(0, 1),
                          optimizer_lr=0.1,
                          max_steps=60,
                          abort_early=True)
            attacking(data_test, modelTrain, Attacker, 1,
                      './attackIms/MnistNet_cw_targeted.pth',
                      is_targeted=True)


            #pgd attack
            # attack = PGDattack(modelTrain.net, early_stop=True, targeted=False,
            #                    eps=ATTACK_PARAMS['mnist']['eps'],
            #                     eps_iter=ATTACK_PARAMS['mnist']['eps_iter'], nb_iter=50)
            # attacking(data_test, modelTrain, attack, 1, './attackIms/mnist_adv_pgd_untargeted.pth', is_targeted=False)

            # attack = PGDattack(modelTrain.net, early_stop=True, targeted=True,
            #                    eps=0.3,
            #                    eps_iter=0.01, nb_iter=50)
            # attacking(data_test, modelTrain, attack, 1, './attackIms/MnistNet_pgd_untargeted2.pth', is_targeted=True)

            # attack = PGDattack(modelTrain.net, early_stop=True, targeted=True,
            #                    eps=0.3,
            #                    eps_iter=0.01, nb_iter=50)
            # attacking(data_test, modelTrain, attack, 1, './attackIms/MnistNet_pgd_targeted.pth', is_targeted=True)
            #


    if args.model == 'resnet_v1':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
        model = ResNet18()
        modelTrain = TrainNN(prefix='Resnet18', net=model)
        if args.resume is None:
            modelTrain.optimizer = torch.optim.SGD(modelTrain.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            modelTrain.scheduler = torch.optim.lr_scheduler.MultiStepLR(modelTrain.optimizer, milestones=[51, 61, 71], gamma=0.1)
            # cpk = [60, 70, 120, 160, 199]
            cpk = [50, 60, 70, 80]
            modelTrain.train(200, trainloader, testloader, writer, cpk)
        else:
            # load vanlia resnet
            # ep = modelTrain.load_checkpoint(args.resume)

            # load robust classifer weights
            ep = 80
            modelTrain.net.load_state_dict(get_robust_classifer_sdic(args.resume))
            modelTrain.test(ep, trainloader)
            modelTrain.test(ep, testloader)
            # Attacker = CW(modelTrain.net,
            #               targeted=True,
            #               c_range=(4, 4.0001),
            #               search_steps=1,
            #               box=(0, 1),
            #               optimizer_lr=0.1,
            #               max_steps=50,
            #               abort_early=True)
            #
            # attacking(testset,
            #           modelTrain,
            #           Attacker,
            #           64,
            #           './attackIms/robust_resnet_cw.pth',
            #           is_targeted=True)

            attack = PGDattack(modelTrain.net,
                               early_stop=True,
                               targeted=True,
                               eps=0.03,
                               eps_iter=0.003,
                               nb_iter=50)

            attacking(testset, modelTrain, attack, 1,
                      './attackIms/robust_pgd_targeted.pth',
                      is_targeted=True)

    if args.model == 'resnet_v2':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
        model = resnet_18_v2()
        modelTrain = TrainNN(prefix='Resnet18_v2', net=model)
        if args.resume is None:
            modelTrain.optimizer = torch.optim.SGD(modelTrain.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
            modelTrain.scheduler = torch.optim.lr_scheduler.MultiStepLR(modelTrain.optimizer, milestones=[60, 120, 180], gamma=0.2)
            cpk = [40, 50, 60, 70, 120, 160, 199]
            modelTrain.train(200, trainloader, testloader, writer, cpk)
        else:
            ep = modelTrain.load_checkpoint(args.resume)
            modelTrain.test(ep, trainloader)
            modelTrain.test(ep, testloader)
            Attacker = CW(modelTrain.net,
                          targeted=True,
                          c_range=(4, 4.0001),
                          confidence=0,
                          search_steps=1,
                          box=(0, 1),
                          optimizer_lr=0.001,
                          max_steps=50,
                          abort_early=False,
                          min_confidence=0
                          )

            attacking(testset,
                      modelTrain,
                      Attacker,
                      1,
                      './attackIms/resnet18_v2_cw.pth',
                      is_targeted=True)

    if args.model == 'resnet_v2_data_aug':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        inputs_box = (min((0 - m) / s for m, s in zip(mean, std)),
                      max((1 - m) / s for m, s in zip(mean, std)))
        print(inputs_box)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
        model = resnet_18_v2()
        modelTrain = TrainNN(prefix='Resnet18_v2_data_aug', net=model)
        if args.resume is None:
            modelTrain.optimizer = torch.optim.SGD(modelTrain.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
            modelTrain.scheduler = torch.optim.lr_scheduler.MultiStepLR(modelTrain.optimizer, milestones=[60, 120, 180], gamma=0.2)
            # cpk = [60, 70, 120, 160, 199]
            cpk = [60, 120, 140, 180, 198]
            modelTrain.train(200, trainloader, testloader, writer, cpk)
        else:
            ep = modelTrain.load_checkpoint(args.resume)
            # modelTrain.test(ep, trainloader)
            # modelTrain.test(ep, testloader)
            Attacker = CW(modelTrain.net,
                          targeted=True,
                          c_range=(4, 4.0001),
                          confidence=4,
                          search_steps=1,
                          box=inputs_box,
                          optimizer_lr=0.001,
                          max_steps=100,
                          abort_early=True)

            attacking(testset,
                      modelTrain,
                      Attacker,
                      1,
                      './attackIms/resnet18_v2_da_cw.pth',
                      is_targeted=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet_v1', required=False)
    parser.add_argument('--resume', type=str, default=None, required=False)

    args = parser.parse_args()
    print(vars(args))
    main()