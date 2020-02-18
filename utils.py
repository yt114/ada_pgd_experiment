import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np

def generate_attack_targets(y_c, class_num=10):
    ''' Randomly choose targets other than the correct lable
    '''
    attack_targets =torch.randint(0, class_num, y_c.shape)
    for i in range(attack_targets.shape[0]):
        if attack_targets[i] == y_c[i]:
            attack_targets[i] = (attack_targets[i] +1) % (class_num)

    return attack_targets

def attacking(data_test, modelTrain, attacker, save_fn, is_targeted=True, max_adv=1000):
    bz = 1
    # perform attack here on test
    success_attack_ims = None
    success_attack_labels = None
    success_original_ims = None

    data_test_loader = DataLoader(data_test, bz , num_workers=1)
    success_attack = 0
    total_ims = 0

    l2norm = 0
    i=0
    confidence_sum = 0
    for x, y in data_test_loader:
        x.to(modelTrain.dev)
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
            y_attack_targets = y_attack_targets.to(modelTrain.dev)
            x_adv = attacker.perturb(x_correct, y_attack_targets)
            print(y_attack_targets)
        else:
            x_adv =attacker.perturb(x_correct, y_correct)

        x_adv.to(modelTrain.dev)

        ind_wrong, _ = modelTrain.test_one_batch(x_adv, y_correct, verbose=1)
        ind_wrong = ind_wrong.squeeze(dim=1)
        success_attack += ind_wrong.shape[0]
        total_ims += ind_correct.shape[0]

        l2norm += (torch.norm(x_adv[ind_wrong].to(modelTrain.dev)-x_correct[ind_wrong].to(modelTrain.dev), 2).sum().item())

        # outputs = F.softmax(modelTrain.net(x_correct[ind_wrong]))
        # confidence, predicted = outputs.max(1)
        # confidence_sum += confidence.cpu().item()

        if success_attack_ims is not None:
            success_original_ims = torch.cat([success_original_ims, x_correct[ind_wrong]])
            success_attack_ims = torch.cat([success_attack_ims, x_adv[ind_wrong]], dim=0)
            success_attack_labels = torch.cat([success_attack_labels, y_correct[ind_wrong]], dim=0)
        else:
            success_attack_ims = x_adv[ind_wrong]
            success_attack_labels = y_correct[ind_wrong]
            success_original_ims = x_correct[ind_wrong]

        if i > 10:
            print(success_attack, total_ims)
            print('l2 norm', l2norm/success_attack)
            print('success attack ratio: ', success_attack/total_ims)

        if success_attack == max_adv:
            break

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

    advs = {'ims': success_original_ims, 'adv': success_attack_ims, 'label': success_attack_labels}
    torch.save(advs, save_fn)
    print(success_attack_ims.shape)
    print(success_attack, total_ims, l2norm/success_attack)


class Hook():
    '''
    For Now we assume the input[0] to last linear layer is a 1*d tensor
    the layerOutput is a list of those tensor value in numpy array
    '''
    def __init__(self, module, layer_id):
            #self.prehook = module.register_forward_pre_hook(self.prehook_fn)
            if layer_id == 1:
                hk = self.hook_fn_1
            else:
                hk = self.hook_fn_2

            self.hook = module.register_forward_hook(hk)
            self.layerOutput = []

    def prehook_fn(self, module, input):
        self.preinput = input

    def hook_fn_1(self, module, input, output):
        feature = input[0].cpu().numpy()
        self.layerOutput.append(feature.flatten())
        pass

    def hook_fn_2(self, module, input, output):
        feature = input[0].cpu().numpy()
        channel_nb = feature.shape[1]
        width = feature.shape[2]
        height = feature.shape[3]
        # print(channel_nb)
        # feature shape (samples, channel, w, h)
        feature = np.reshape(feature, (1, channel_nb, width*height))
        feature = np.sum(feature, axis=2) / (width*height)
        # feature = np.max(feature, axis=2)
        # print(feature.shape)
        self.layerOutput.append(feature.flatten())
        # self.input = input
        # self.output = output
        pass

    def close(self):
        # self.prehook.remove()
        self.hook.remove()

def getLayerOutput(ds, model, hook, ys, outs=None):
    ''' Get the layer outputs, real data class, NN outputs for  GMM train/testing
    Args:
        ds (torch.tensor): dataset of data
        model (torch.module):
        hook (Hook): self-defined hook class
        ys (None/np.array): record real data class of shape (num_samples, )
            if none, no recording
        outs (None/np.array): record  nn models' ouput (num_samples, class_nums)
            if none, no recording

    Returns: None
    '''
    print(len(ds))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
    id = 0
    model.eval()
    correct = 0
    tot = 0
    confidence_sum = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dl):
            if ys is not None:
                ys[id] = targets.item()
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = F.softmax(model(inputs))
            if outs is not None:
                outs[id, :] = outputs.cpu().numpy()
            confidence, predicted = outputs.max(1)
            confidence_sum += confidence.cpu().item()
            id += 1
            if predicted.item() == targets.item():
                correct += 1
            tot+=1

            if batch_idx % 100 == 0:
                print('batch ind: ', batch_idx)
                print('avg confidence: ', confidence_sum/tot)

    hook.close()
    print('acc: {}/{} = {:.2f}'.format(correct, tot, correct/tot))
    return


def estimate_confusion_prob(modelTrain, dl):
    modelTrain.net.eval()

    y_true = None
    y_pred = None
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dl):
            inputs, targets = inputs.to(modelTrain.dev), targets.to(modelTrain.dev)
            outputs = modelTrain.net(inputs)
            _, predicted = outputs.max(1)
            if y_true is None:
                y_true = targets.cpu().squeeze().numpy()
                y_pred = predicted.cpu().numpy()
            else:
                y_true = np.concatenate([y_true, targets.cpu().numpy()], axis=0)
                y_pred = np.concatenate([y_pred, predicted.cpu().numpy()], axis=0)

    # return confusionProbability(y_true, y_pred)

    return y_true, y_pred
