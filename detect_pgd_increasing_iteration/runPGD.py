from modelZoo import LeNet5
from NNmodels import TrainNN
import torch
import torchvision
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from pgdAttack import PGDattack
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt


start_it = -1
end_it = 0
cur_it = 0

model_nm = ''
previous_attack_im = ''
cur_attack_im_fn = ''

log_file = './PGD_attack_data.txt'
save_examples = 10
step_it = 2
max_adv_num = 1000

ATTACK_PARAMS = {
    'mnist': {'eps': 0.003, 'eps_iter': 0.0005},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010}
}

def further_perturb(dl, modelTrain, attacker, save_fn):
    ''' Now only for untargeted attack
    :param data_test:
    :param modelTrain:
    :param attacker:
    :param save_fn:
    :param is_targeted:
    :param max_adv:
    :return:
    '''
    success_attack_ims = []
    success_labels = []
    success_original_ims = []

    success_attack = 0
    total_ims = 0
    l2norm = 0
    confidence_sum = 0

    saved_examples = 0
    for x_clean, x, y in dl:
        modelTrain.net.eval()
        yvar = y.clone().detach()
        x = x.to(modelTrain.dev)
        y = y.to(modelTrain.dev)
        x_clean = x_clean.to(modelTrain.dev)
        x_adv = attacker.perturb(x, y)
        x_adv = x_adv.to(modelTrain.dev)
        with torch.no_grad():
            outputs = F.softmax(modelTrain.net(x_adv))
            outputs = outputs.cpu()
        confidence, predicted = outputs.max(1)

        ind_wrong = (~predicted.eq(yvar)).nonzero()
        # ind_correct = predicted.eq(yvar).nonzero()
        print('debug: y.shape[0]: {}, equal num: {}'.format(yvar.shape[0],
                                                            predicted.eq(yvar).sum().item()))

        correct = (yvar.shape[0] - predicted.eq(yvar).sum().item())
        success_attack += correct
        total_ims += yvar.shape[0]
        if ind_wrong.shape[0] != 0:
            confidence_sum += confidence[ind_wrong].cpu().sum().item()
            l2norm += (
                torch.norm(x_adv[ind_wrong].to(modelTrain.dev) - x_clean[ind_wrong].to(modelTrain.dev), 2).sum().item())

            success_original_ims.append(x_clean[ind_wrong])
            success_attack_ims.append(x_adv[ind_wrong])
            success_labels.append(y[ind_wrong])

            i = 0
            while (i < x_adv.shape[0]) and (saved_examples < save_examples):
                tem_x = x_clean[i]
                tem_y = x_adv[i]
                im_normal = x_clean[i].cpu().clone().detach()
                im_adv = x_adv[i].cpu().clone().detach()
                img = torchvision.utils.make_grid([im_normal, im_adv])
                plt.figure()
                npimg = img.numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                print(cur_it)
                plt.title("right: it{}, adversarial image (confidence:{:.2f})".format(cur_it, confidence[i].item()))
                plt.savefig('./examples/it{}_num{}.jpg'.format(cur_it, saved_examples))
                plt.clf()
                saved_examples += 1
                i += 1

            if (success_attack > 0) and success_attack % 10 == 0:
                print('total images number: {},'.format(total_ims),
                      'success attack num: {}'.format(success_attack),
                      'attack success: {:.4f},'.format(success_attack / total_ims),
                      'l2 norm dist: {:.4f},'.format(l2norm / success_attack),
                      'confidence: {:.4f},'.format(confidence_sum / success_attack))
        if success_attack > max_adv_num:
            break
    success_original_ims = torch.cat(success_original_ims, dim=0)
    success_attack_ims = torch.cat(success_attack_ims, dim=0)
    success_labels = torch.cat(success_labels, dim=0)
    succ_dict = {'im': success_original_ims, 'adv': success_attack_ims, 'label': success_labels}

    torch.save(succ_dict, save_fn)
    sample = open(log_file, 'a')
    print('iteration {},'.format(cur_it),
            'total images number: {},'.format(total_ims),
          'success attack num: {},'.format(success_attack),
          'attack success: {:.4f},'.format(success_attack / total_ims),
          'l2 norm dist: {:.4f},'.format(l2norm / success_attack),
          'confidence: {:.4f},'.format(confidence_sum / success_attack), file=sample)

    pass

def First_PGD_perturb(dl, modelTrain, attacker, save_fn):
    ''' Now only for untargeted attack
    :param data_test:
    :param modelTrain:
    :param attacker:
    :return:
    '''
    success_attack_ims = []
    success_labels = []
    success_original_ims = []

    success_attack = 0
    total_ims = 0
    l2norm = 0
    confidence_sum = 0

    saved_examples = 0
    for x, y in dl:
        modelTrain.net.eval()
        yvar = y.clone().detach()
        x_clean = x.clone().detach()

        x = x.to(modelTrain.dev)
        y = y.to(modelTrain.dev)

        x_adv = attacker.perturb(x, y)
        x_adv = x_adv.to(modelTrain.dev)

        with torch.no_grad():
            outputs = F.softmax(modelTrain.net(x_adv))
            outputs = outputs.cpu()
        confidence, predicted = outputs.max(1)

        ind_wrong = (~predicted.eq(yvar)).nonzero()
        # ind_correct = predicted.eq(yvar).nonzero()
        print('debug: y.shape[0]: {}, equal num: {}'.format(yvar.shape[0],
                                                             predicted.eq(yvar).sum().item()))

        correct =(yvar.shape[0] - predicted.eq(yvar).sum().item())
        success_attack += correct
        total_ims += yvar.shape[0]
        if ind_wrong.shape[0] != 0:
            confidence_sum += confidence[ind_wrong].cpu().item()
            l2norm += (torch.norm(x_adv[ind_wrong].to(modelTrain.dev) - x_clean[ind_wrong].to(modelTrain.dev), 2).sum().item())

            success_original_ims.append(x_clean[ind_wrong])
            success_attack_ims.append(x_adv[ind_wrong])
            success_labels.append(y[ind_wrong])

            if saved_examples < save_examples:
                for i in range(0, x_adv.shape[0]):
                    im_normal = np.squeeze(x_clean, axis=0).cpu().clone().detach()
                    im_adv = np.squeeze(x_adv, axis=0).cpu().clone().detach()
                    img = torchvision.utils.make_grid([im_normal, im_adv])
                    print(confidence)
                    plt.figure()
                    npimg = img.numpy()
                    plt.imshow(np.transpose(npimg, (1, 2, 0)))
                    plt.title("right: it{}, adversarial image (confidence:{:.2f})".format(0, confidence.item()))
                    plt.savefig('./examples/it{}_num{}.jpg'.format(0, saved_examples))
                    saved_examples += 1

            if (success_attack>0) and success_attack % 10 == 0:
                print('total images number: {},'.format(total_ims),
                      'success attack num: {}'.format(success_attack),
                      'attack success: {:.4f},'.format(success_attack/total_ims),
                      'l2 norm dist: {:.4f},'.format(l2norm/success_attack),
                      'confidence: {:.4f},'.format(confidence_sum/success_attack))

        if success_attack > max_adv_num:
            break
    success_original_ims = torch.cat(success_original_ims, dim=0)
    success_attack_ims = torch.cat(success_attack_ims, dim=0)
    success_labels = torch.cat(success_labels, dim=0)
    succ_dict = {'im':success_original_ims, 'adv':success_attack_ims, 'label':success_labels}

    torch.save(succ_dict, save_fn)
    torch.save(succ_dict, save_fn)
    sample = open(log_file, 'w')
    print('total images number: {},'.format(total_ims),
          'success attack num: {}'.format(success_attack),
          'attack success: {:.4f},'.format(success_attack / total_ims),
          'l2 norm dist: {:.4f},'.format(l2norm / success_attack),
          'confidence: {:.4f},'.format(confidence_sum / success_attack), file=sample)
    pass

def main():
    global start_it
    global end_it
    global cur_it
    global saved_examples
    model_nm = args.model
    start_it = args.start_it
    end_it = args.end_it
    cur_it = start_it + 1

    previous_attack_im = './attackIms/{}_it_{}.pth'.format(model_nm, start_it)
    cur_attack_im_fn  = './attackIms/{}_it_{}.pth'.format(model_nm, end_it)


    if model_nm == 'MNIST':
        data_train = MNIST('./data/mnist', download=True,
                           transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

        data_test = MNIST('./data/mnist', train=False, download=True,
                          transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
        print("Run PGD on Mnist test dataset of size ", len(data_test))

        model = LeNet5()
        prefix = 'Mnist'
        modelTrain = TrainNN(prefix, net=model)
        modelTrain.load_checkpoint(args.resume)

    if start_it == -1:  # craft samples at decision boundary
        test_loader = DataLoader(data_test, shuffle=True)

        attacker = PGDattack(modelTrain.net,
                           early_stop=True,
                           targeted=False,
                           eps=ATTACK_PARAMS['mnist']['eps'],
                           eps_iter=ATTACK_PARAMS['mnist']['eps_iter'],
                           nb_iter=100)

        First_PGD_perturb(test_loader, modelTrain, attacker, cur_attack_im_fn)

    while cur_it < end_it:
        previous_attack_im = './attackIms/{}_it_{}.pth'.format(model_nm, start_it)
        cur_attack_im_fn = './attackIms/{}_it_{}.pth'.format(model_nm, cur_it)
        print("loading from previous attack ims: ", previous_attack_im)
        adv_ds = torch.load(previous_attack_im)
        ims = adv_ds['im'].cpu()
        ims = ims.squeeze(axis=1)
        print(ims.shape)
        advs = adv_ds['adv'].cpu()
        advs = advs.squeeze(axis=1)
        print(advs.shape)
        labels = adv_ds['label'].cpu()
        labels = labels.squeeze(axis=1)
        print(labels.shape)
        ds = torch.utils.data.TensorDataset(ims, advs, labels)
        test_loader = DataLoader(ds, batch_size=128, shuffle=False)
        attacker = PGDattack(modelTrain.net,
                         early_stop=False,
                         targeted=False,
                         eps=ATTACK_PARAMS['mnist']['eps'],
                         eps_iter=ATTACK_PARAMS['mnist']['eps_iter'],
                         nb_iter=step_it)
        further_perturb(test_loader, modelTrain, attacker, cur_attack_im_fn)
        start_it = cur_it
        print(cur_it)
        cur_it += step_it
        save_examples = 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None, required=False)
    parser.add_argument('--start_it', type=int)
    parser.add_argument('--end_it', type=int, required=True)

    args = parser.parse_args()
    print(vars(args))

    main()
