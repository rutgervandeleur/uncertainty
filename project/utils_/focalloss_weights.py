import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class FocalLoss_AuxOutput(nn.Module):
    def __init__(self, num_classes, device, gamma=0, alpha=None, size_average=True, weights = None):
        super(FocalLoss_AuxOutput, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.weights = weights
        self.device = device
        self.Gauss = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(num_classes), torch.eye(num_classes))

    # Takes T samples from the logits, by corrupting the network output with
    # Gaussian noise with variance determined by the networks auxiliary
    # outputs. 
    # As in "What uncertainties do we need in Bayesian deep learning for
    # computer vision?", equation (12), first part.
    def sample_logits(self, T, input, variance):
        
        # Take T samples from the Gaussian distribution
        epsilon = self.Gauss.sample([input.shape[0], T]).to(self.device)

        # Go from shape: [batch x num_classes] -> [batch x T x num_classes]
        sigma = variance[:, None, :].repeat(1, T, 1)
        f = input[:, None, :].repeat(1, T, 1)

        # Multiply Gaussian noise with variance, and add to the prediction
        x_t =  f + sigma * epsilon 
        return x_t

    # Calculates the stochastic loss according to
    # "What uncertainties do we need in Bayesian deep learning for
    # computer vision?", equation (12), second part.
    def calc_stochastic_loss(self, T, x_t, target):

        # Select values in ground_truth class column
        x_tc = torch.gather(x_t, 2, target[:, None, :].repeat(1, T, 1)).squeeze(2)

        loss = torch.log( torch.exp( x_tc - torch.log( torch.exp( x_t ).sum(dim=2) ) ).mean(dim=1) )
        return loss


    # Variance is the individual variance for each class in each prediction
    def forward(self, input, variance, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # n,c,h,w => n,c,h*w
            input = input.transpose(1,2)    # n,c,h*w => n,h*w,c
            input = input.contiguous().view(-1,input.size(2))   # n,h*w,c => n*h*w,c
        target = target.view(-1,1)

        # T = number of logit samples
        T = 20
        x_t = self.sample_logits(T, input, variance)
# --------------- Implementation v1 -------------------
        #stochastic_loss = self.calc_stochastic_loss(T, x_t, target)
        #pt = stochastic_loss.exp()
        #logpt = stochastic_loss
# -----------------------------------------------------
        # Implementation V2
        logpt = F.log_softmax(x_t.mean(dim=1), dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp()) 
    
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * variable(at)
        
        loss = -1 * (1-pt)**self.gamma * logpt

        # Multiply weights with the loss.
        # Dependent on the ground-truth label, the corresponding weight is mutliplied with the loss
        if self.weights is not None:
            w = self.weights.repeat(target.shape[0],1).gather(1,target)
            w = w.view(-1)
            loss = loss * w

        if self.size_average: return loss.mean()
        else: return loss.sum()

           
    
class FocalLoss(nn.Module):
    """
    gamma: {0, 5}, rate of down-weighting loss of easy examples
    alpha: [{0, 1}] weights for each class
    size_average: Whether to average the loss over the whole batch
    """
    def __init__(self, gamma=0, alpha=None, size_average=True, weights = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.weights = weights

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        target = target.to('cuda')
        logpt = F.log_softmax(input, dim=1).to('cuda')
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        
        if self.weights is not None:
            ic(target)
            w = self.weights.repeat(target.shape[0],1).gather(1,target)
            w = w.view(-1)
            loss = loss * w
        
        if self.size_average==True: return loss.mean()
        else: return loss

#-------------------------------------------------
        # Old auxiliary loss calculation
        #ctilde_sum = 0
        #for ctilde in x_t[0, 0]:
        #    ctilde_sum += torch.exp(ctilde)
        #print(ctilde_sum, 'ctilde_sum')
        #log_ctilde_sum = torch.log(ctilde_sum)
        #print(log_ctilde_sum)
        #exp_loss = torch.exp(x_t[0, 0, target[0]] - log_ctilde_sum)
        #print(exp_loss)
        #
        ## torch.log(torch.exp(x_t[0, :, target[0]]- torch.log(torch.exp(x_t[0, :]).sum(dim=2))).mean(dim=1))
        #sum_c_exp_x_t = torch.exp(x_t).sum(dim=2)
        #print('sum_c_exp_x_t', sum_c_exp_x_t)
        #print(sum_c_exp_x_t.shape)
        #print(x_t.shape, target[:, None, :].repeat(1, T, 1).shape)
        #print('x_t.shape, target.shape')
        ## Select values in ground_truth class column
        #x_t_c = torch.gather(x_t, 2, target[:, None, :].repeat(1, T, 1))
        #print('x_t_c.shape', x_t_c.shape)

        #print(x_t[0], target[0], x_t_c[0])
        ## X_tc - log ( sum_c( exp (x_t) ))
        #x_t_c_min_log_sum_c_exp_x_t = x_t_c.squeeze(2) - torch.log(sum_c_exp_x_t)
        #print(x_t_c_min_log_sum_c_exp_x_t.shape)
        #print('x_t_c_min_log_sum_c_exp_x_t.shape')
        #stochastic_loss = torch.log( torch.exp(x_t_c_min_log_sum_c_exp_x_t).mean(dim=1) )
        ## stochastic_loss =  torch.exp(x_t_c_min_log_sum_c_exp_x_t).mean(dim=1)
        
#-------------original implementation----------------------
        #logpt = F.log_softmax(input, dim=1)
        #logpt = logpt.gather(1,target)
        #logpt = logpt.view(-1)
        #pt = variable(logpt.data.exp())
        #---------------------------------------------------------

