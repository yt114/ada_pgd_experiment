import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from modelZoo import ResNet18
from NNmodels import TrainNN
from collections import OrderedDict
from pgdAttack import PGDattack
from utils import attacking, Hook, getLayerOutput
import numpy as np
from utils import estimate_confusion_prob

def get_robust_classifer_sdic(resumepath):
    ''' robust trained classifer is wrapped in DataParallel module.
        the keys of state_dict is preceeded by 'module'
    :param resumepath:
    :return:
    '''
    # original saved file with DataParallel
    state_dict = torch.load(args.resume)
    # create new OrderedDict that does not contain `module.`

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict



def main():
    prefix = 'robust'
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=256,
                                             shuffle=False,
                                             num_workers=4)

    model = ResNet18()
    modelTrain = TrainNN(prefix='Resnet18', net=model)
    if args.robust:
        s = get_robust_classifer_sdic(args.resume)
    else:
        s = torch.load(args.resume)

    modelTrain.net.load_state_dict(s)
    modelTrain.test(0, testloader)

    adv_set, val_set = torch.utils.data.random_split(testset,
                                                       [5000, 5000])
    torch.save(adv_set, './adv_set.pth')
    torch.save(val_set, './val_set.pth')


    adv_set = torch.load('./adv_set.pth')
    val_set = torch.load('./val_set.pth')
    attack = PGDattack(modelTrain.net,
                       early_stop=True,
                       targeted=True,
                       eps=0.05,
                       eps_iter=0.001,
                       nb_iter=50)

    attacking(adv_set, modelTrain, attack,
              './attackIms/{}_pgd_targeted.pth'.format(prefix))

    adv_ds = torch.load('./attackIms/{}_pgd_targeted.pth'.format(prefix))
    ims = adv_ds['ims'].cpu()
    advs = adv_ds['adv'].cpu()
    labels = adv_ds['label'].cpu()
    print(ims.shape, labels.shape)

    # print('*'*10, 'getting training dataset intermediate layer outputs', '*'*10)
    # targetlayer = modelTrain.net._modules['linear']
    # hooker = Hook(targetlayer, layer_id=1)
    # ys = np.zeros(len(trainset))  # record real data class
    # getLayerOutput(trainset, model, hooker, ys, outs=None)
    # lo = np.squeeze(np.array(hooker.layerOutput))
    # print(lo.shape)
    # np.save('./layeroutputs/{}_train_data.npy'.format(prefix),
    #         lo)  # intermediate layer outputs
    # np.save('./layeroutputs/{}_train_label.npy'.format(prefix),
    #         ys) # real data class


    print('getting adversaraial image intermediate outputs', '*'*10)
    ds = torch.utils.data.TensorDataset(advs, labels)
    targetlayer = modelTrain.net._modules['linear']
    hooker = Hook(targetlayer, layer_id=1)
    outputs = np.zeros((len(ds), 10))
    getLayerOutput(ds, model, hooker, None, outputs)
    lo = np.squeeze(np.array(hooker.layerOutput))
    print(lo.shape)
    np.save('./layeroutputs/{}_adv_data.npy'.format(prefix),
            lo)  # intermediate layer outputs
    np.save('./layeroutputs/{}_adv_outputs.npy'.format(prefix),
            outputs) # NN model outputs

    print('getting original image intermediate outputs', '*'*10)
    ds = torch.utils.data.TensorDataset(ims, labels)
    targetlayer = modelTrain.net._modules['linear']
    hooker = Hook(targetlayer, layer_id=1)
    outputs = np.zeros((len(ds), 10))
    getLayerOutput(ds, model, hooker, None, outputs)
    lo = np.squeeze(np.array(hooker.layerOutput))
    print(lo.shape)
    np.save('./layeroutputs/{}_test_data.npy'.format(prefix),
            lo)  # intermediate layer outputs
    np.save('./layeroutputs/{}_test_outputs.npy'.format(prefix),
            outputs) # NN model outputs


    # estimate confusion matrix on validation set
    print('*'*10, 'compute confusion prob', '*'*10)
    print('validation set size: ', len(val_set))
    valloader = torch.utils.data.DataLoader(val_set,
                                             batch_size=256,
                                             shuffle=False,
                                             num_workers=4)
    y, y_pred = estimate_confusion_prob(modelTrain, valloader)
    np.save('./confusionProbMat/{}_val_y_true'.format(prefix), y)
    np.save('./confusionProbMat/{}_val_y_pred'.format(prefix), y_pred)
    print('save to')
    print('./confusionProbMat/{}_val_y_true'.format(prefix))
    print('./confusionProbMat/{}_val_y_pred'.format(prefix))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        required=False)

    parser.add_argument('--robust',
                        default=False,
                        dest='robust',
                        action='store_true'
                        )


    args = parser.parse_args()
    print(vars(args))
    main()

