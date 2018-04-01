import warnings
import math
from operator import mul
from functools import reduce

import torch

def eq_nll_loss(input, target, eq_weight, weight=None, size_average=True, ignore_index=-100, reduce=True, epsilon=1e-15):
    r"""The negative log likelihood loss.

    See :class:`~torch.nn.NLLLoss` for details.

    Args:
        input: :math:`(N, C)` where `C = number of classes` or `(N, C, H, W)`
            in case of 2D - Loss
        target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. If size_average
            is False, the losses are summed for each minibatch. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average is
            True, the loss is averaged over non-ignored targets. Default: -100

    Example::

        >>> # input is of size N x C = 3 x 5
        >>> input = autograd.Variable(torch.randn(3, 5))
        >>> # each element in target has to have 0 <= value < C
        >>> target = autograd.Variable(torch.LongTensor([1, 0, 4]))
        >>> output = F.nll_loss(F.log_softmax(input), target)
        >>> output.backward()
    """
    input_clipped = torch.clamp(input, epsilon, 1-epsilon)

    #m = input.shape[0]

    #print(input.shape)
    #print(input_clipped.shape)
    #print(target.shape)
    #print(eq_weight.shape)
    #print('-----')
    #print(input)
    #print('-----')
    #print(input_clipped)
    #print('-----')
    #print(target)
    #print('-----')
    #print(eq_weight)



    loss = -(target.float()*torch.log(input_clipped.float()) + (1 - target.float())*(torch.log(1 - input_clipped.float())))
    #print(loss)


    loss = loss*eq_weight.float()

    return torch.sum(loss)