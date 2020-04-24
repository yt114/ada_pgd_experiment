import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import os

class AverageMeter(object):
    """Computes and stores the average and current value
        copied from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L165-L171
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class TrainNN(object):
    '''A class wraper for neural network training, testing, saving
    '''
    def __init__(self, prefix, net, criterion=None, optimizer=None, scheduler=None):
        super().__init__()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dev = torch.device("cuda")
        print('dev: {} \n'.format(self.dev))
        self.net = net.to(self.dev)
        self.prefix = prefix
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.net.parameters(),  weight_decay=10e-7)
        else:
            self.optimizer = optimizer
        # set scheduler outside of class definition scope


    def train_one_epoch(self, epoch, dataLoader, writer=None):
        '''
        :param writer: torch.utils.tensorboard.SummaryWriter
        :return:
        '''
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        # top5 = AverageMeter('Acc@5', ':6.2f')

        self.net.train()
        start = time.time()

        print('\nEpoch: %d' % epoch)
        self.net.train()

        for batch_idx, (inputs, targets) in enumerate(dataLoader):
            data, targets = inputs.to(self.dev), targets.to(self.dev)
            self.optimizer.zero_grad()
            outputs = self.net(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), inputs.shape[0])

            _, predicted = outputs.max(1)
            acc = predicted.eq(targets).sum().item() / targets.shape[0]
            top1.update(acc, targets.shape[0])

        print('Time: {:.2f} sec'.format(time.time()-start), end='\t')
        print('Training epoch: {}'.format(epoch), end='\t')
        print('Average acc: {:.3f}'.format(top1.avg), end='\t')
        print('Average loss: {:.3f}'.format(losses.avg), end='\n')

        if writer is not None:
            writer.add_scalar('{}/Accuracy/train'.format(self.prefix), top1.avg, epoch)
            writer.add_scalar('{}/Loss/train'.format(self.prefix), losses.avg, epoch)

        return
    def train(self, epoch, traindl, testdl, writer=None, cpk=None):
        for i in range(epoch):
            self.train_one_epoch(i, traindl, writer)
            self.test(i, testdl, writer)
            if cpk is not None and i in cpk:
                self.save_checkpoint(i)
            if self.scheduler is not None:
                self.scheduler.step()

    def test_one_batch(self, inputs, targets, verbose=1):
        self.net.eval()
        with torch.no_grad():
            inputs, targets = inputs.to(self.dev), targets.to(self.dev)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            _, predicted = outputs.max(1)
            acc = predicted.eq(targets).sum().item() / targets.shape[0]

            if verbose > 0:
                print('Average acc: {:.3f}'.format(acc), end='\t')
                print('Average loss: {:.3f}'.format(loss.item()), end='\n')
                print(predicted, torch.max(F.softmax(outputs)))
                print(F.softmax(outputs))
        return (~predicted.eq(targets)).nonzero(), predicted.eq(targets).nonzero()

    def test(self, epoch, dataloader, writer=None):
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        # top5 = AverageMeter('Acc@5', ':6.2f')

        self.net.eval()
        start = time.time()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.dev), targets.to(self.dev)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                losses.update(loss.item(), inputs.shape[0])

                _, predicted = outputs.max(1)
                acc = predicted.eq(targets).sum().item() / targets.shape[0]
                top1.update(acc, targets.shape[0])


        print('Time: {:.2f} sec'.format(time.time()-start), end='\t')
        print('Testing epoch: {}'.format(epoch), end='\t')
        print('Average acc: {:.3f}'.format(top1.avg), end='\t')
        print('Average loss: {:.3f}'.format(losses.avg), end='\n')

        if writer is not None:
            writer.add_scalar('{}/Accuracy/test'.format(self.prefix), top1.avg, epoch)
            writer.add_scalar('{}/Loss/test'.format(self.prefix), losses.avg, epoch)
        return

    def save_checkpoint(self, epoch):
        state = {
            'net': self.net.state_dict(),
            'epoch': epoch+1,
            'optimizer': self.optimizer.state_dict(),
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        print(self.prefix)
        torch.save(state, './checkpoint/{}_{}.pth'.format(self.prefix, epoch))
        print('Saving checkpoint..')
        return

    def load_checkpoint(self, ckp):
        if os.path.isfile(ckp):
            print("=> loading checkpoint '{}'".format(ckp))
            checkpoint = torch.load(ckp)
            start_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckp, checkpoint['epoch']))

            return start_epoch
        else:
            raise FileNotFoundError(ckp)
