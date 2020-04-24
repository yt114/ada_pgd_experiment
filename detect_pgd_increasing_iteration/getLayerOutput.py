import argparse
from torchvision.datasets.mnist import MNIST
from NNmodels import TrainNN
import torchvision.transforms as transforms
import numpy as np
from modelZoo import LeNet5
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
model_nm = ''
it = 0
ims_pth = ''


class Hook():
    '''
    For Now we assume the input[0] to last linear layer is a 1*d tensor
    the layerOutput is a list of those tensor value in numpy array
    '''
    def __init__(self, module, layer_id):
        if layer_id == 1:
            hk = self.hook_fn_1
            self.hook = module.register_forward_hook(hk)
            self.layerOutput = []

    def hook_fn_1(self, module, input, output):
        print(input[0].shape)
        feature = input[0].clone().detach().cpu()
        self.layerOutput.append(feature)
        pass

    def close(self):
        self.hook.remove()


def extract_feature(dl, net, hook, save_fn):
    ''' Get the layer outputs, real data class,
        NN outputs for generative model train/testing
    '''
    correct = 0
    tot = 0
    labels = []
    softmax_outputs = []

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dl):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = F.softmax(net(inputs))

            _, predicted = outputs.max(1)

            correct += predicted.eq(targets).sum().item()
            tot += targets.shape[0]

            labels.append(targets.clone().detach().cpu())
            softmax_outputs.append(outputs.clone().detach().cpu())

    hook.close()

    print('test acc: {}/{} = {:.2f}'.format(correct, tot, correct/tot))
    labels = torch.cat(labels, dim=0)
    softmax_outputs = torch.cat(softmax_outputs, dim=0)
    extracted_feature = hook.layerOutput
    extracted_feature = torch.cat(extracted_feature, dim=0)

    labels = labels.numpy()
    softmax_outputs = softmax_outputs.numpy()
    extracted_feature = extracted_feature.numpy()
    save_dict = {'feature':extracted_feature,
                 'softmax':softmax_outputs,
                 'label':labels}

    print('save to file: ', save_fn)
    np.save('./features/{}.npy'.format(save_fn), save_dict)
    pass

def main():
    global model_nm
    global it
    global ims_pth

    model_nm = args.model
    it = args.it

    if args.model == 'MNIST':
        model = LeNet5()
        prefix = 'MnistNet'
        print(model)

    modelTrain = TrainNN(prefix, net=model)
    modelTrain.load_checkpoint(args.resume)

    if args.train_data == "True":
        if model_nm == "MNIST":
            data_train = MNIST('./data/mnist', download=True,
                               transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
            dl = DataLoader(data_train, batch_size=128, shuffle=False)
            target_layer = (modelTrain.net._modules['fc'])._modules['f7']
            hooker = Hook(target_layer, 1)
            fn = '{}_train_ims_out'.format(model_nm)
            extract_feature(dl, modelTrain.net, hooker, fn)


    ims_pth = './attackIms/{}_it_{}.pth'.format(model_nm, it)
    print("loading dataset from ims: ", ims_pth)

    adv_ds = torch.load(ims_pth)
    if it == 1:
        clean_ims = adv_ds['im'].cpu().squeeze(axis=1)
        print('load clean ims of shape:', clean_ims.shape)

    adv_ims = adv_ds['adv'].cpu().squeeze(axis=1)
    print('load crafted images of shape:', adv_ims.shape)

    labels = adv_ds['label'].cpu().squeeze(axis=1)
    print('load lable of shape:', labels.shape)

    if it == 1:
        ds = torch.utils.data.TensorDataset(clean_ims, labels)
        dl = DataLoader(ds, batch_size=128, shuffle=False)
        target_layer = (modelTrain.net._modules['fc'])._modules['f7']
        hooker = Hook(target_layer, 1)
        fn = '{}_clean_ims_out'.format(model_nm)
        extract_feature(dl, modelTrain.net, hooker, fn)

    ds = torch.utils.data.TensorDataset(adv_ims, labels)
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    target_layer = (modelTrain.net._modules['fc'])._modules['f7']
    hooker = Hook(target_layer, 1)
    fn = '{}_adv_ims_it_{}_out'.format(model_nm, it)
    extract_feature(dl, modelTrain.net, hooker, fn)

    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        help='Chose which neural net from resnet, wide_resnet, etc.',
                        default='resnet'
                        )

    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        required=False,
                        help='path of pretrained NN model'
                        )


    parser.add_argument('--it',
                        type=int
                        )

    parser.add_argument('--train_data',
                        type=str,
                        default='False'
                        )
    args = parser.parse_args()
    main()