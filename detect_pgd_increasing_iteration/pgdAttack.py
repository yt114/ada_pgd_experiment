import torch
from torch import nn

'''script adopted from:
    https://github.com/BorealisAI/advertorch
    
    In addition, we implement early stop method (Feinman et al, 2017)
    BIM-A, which stops iterating as soon as miclassification is achieved
    (‘at the decision boundary’)
    Paper: https://arxiv.org/pdf/1703.00410.pdf
'''

class PGDattack(object):
    def __init__(self, model, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=False, clip_min=0., clip_max=1.,
            targeted=False, early_stop=False):
        """
        The projected gradient descent attack (Madry et al, 2017).
        The attack performs nb_iter steps of step size eps_iter, while always staying
        within eps from the initial point.
        Paper: https://arxiv.org/pdf/1706.06083.pdf



        :param model: forward pass function-NN.
        :param loss_fn: loss function.
        :param eps: maximum distortion.
        :param nb_iter: number of iterations.
        :param eps_iter: attack step size.
        :param rand_init: (optional bool) random initialization.
        :param clip_min: mininum value per input dimension.
        :param clip_max: maximum value per input dimension.
        :param targeted: if the attack is targeted.
        :param early_stop: turn on BIM-A method when true.
                            Require the batch size = 1
        """
        self.model = model
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter

        self.rand_init = rand_init
        if self.rand_init:
            raise NotImplementedError('PGD rand init no implemented')

        self.targeted = targeted
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.early_stop = early_stop
        pass


    def perturb(self, x, y):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        if self.early_stop:
            assert x.shape[0] == 1

        self.model.eval()
        xvar, yvar = x.clone().detach(), y.clone().detach()

        delta = torch.zeros_like(xvar)
        delta.requires_grad_()

        # print(xvar.device, delta.device)
        for i in range(self.nb_iter):
            outputs = self.model(xvar + delta)

            # early stop
            if self.early_stop:
                _, predicted = outputs.data.max(1)
                if self.targeted and torch.equal(predicted, yvar):
                    #print('PGD stop after targeted misclassification achieved')
                    break
                if self.targeted is False and not torch.equal(predicted, yvar):
                    #print('PGD stop after untargeted misclassification achieved')
                    # print(predicted, yvar)
                    break

            loss = self.loss_fn(outputs, yvar)
            # for untargeted attack, maximize the CE loss
            # targeted attack, minimize loss
            if self.targeted:
                loss = - loss

            loss.backward()
            # any changes to torch.tensor.data wouldn't be tracked by autograd
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + self.eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, max=self.eps)

            # clamp xadv = xvar.data + delta.data so that it is in pixel value range
            delta.data = torch.clamp(xvar.data + delta.data, self.clip_min, self.clip_max
                               ) - xvar.data

            delta.grad.data.zero_()

        x_adv = torch.clamp(xvar.data + delta.data, self.clip_min, self.clip_max)
        return x_adv.data