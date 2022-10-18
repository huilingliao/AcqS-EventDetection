import numpy as np
import torch
from torch import nn
from torch.autograd import grad as torchgrad

from collections import Counter

def unique(list1):
    return list(Counter(list1).keys())


def HMCwithAccept(energy,x,length,steps,epsilon,burnin,n_gap):
    shape = [i if no==0 else 1 for no,i in enumerate(x.shape)]
    def grad(z):
        return torchgrad(energy(z),z,grad_outputs=z.new_ones(z.shape[0]))[0]
    torch.set_grad_enabled(False)
    E = energy(x)
    torch.set_grad_enabled(True)
    g = grad(x.requires_grad_())
    torch.set_grad_enabled(False)
    g = g.detach()
    if torch.norm(g, 2) > 0:
        g = g / torch.norm(g, 2)

    x_list = []
    accept_list = []

    for l in range(length):
        p = x.new_empty(size=x.size()).normal_()
        H = ((0.5*p*p).reshape(p.shape[0], -1).sum(dim=1) + E)
        xnew = x
        gnew = g
        for _ in range(steps):
            p = p- epsilon* gnew/2.
            xnew = (xnew + epsilon * p)
            torch.set_grad_enabled(True)
            gnew = grad(xnew.requires_grad_())
            torch.set_grad_enabled(False)
            xnew = xnew.detach()
            gnew = gnew.detach()
            if torch.norm(gnew, 2) > 0:
                gnew = gnew / torch.norm(gnew, 2)
            p = p- epsilon* gnew/2.
        Enew = energy(xnew)
        Hnew = (0.5*p*p).reshape(p.shape[0], -1).sum(dim=1) + Enew
        diff = H-Hnew
        accept = (diff.exp() >= diff.uniform_()).to(x)

        E = accept*Enew + (1.-accept)*E
        acceptMask = accept.reshape(shape)
        x = acceptMask*xnew + (1.-acceptMask)*x
        g = acceptMask*gnew + (1.-acceptMask)*g
        x_list.append(x)
        accept_list.append(accept)

    torch.set_grad_enabled(True)
    return x_list[burnin::n_gap], accept_list[burnin::n_gap]


def forConstraints(x, p, bounds):
    dim = len(bounds[0])
    flag = torch.any(torch.logical_and(x > bounds[1], x < bounds[0]))
    while flag:
        adj_idx = torch.logical_or(x > bounds[1], x < bounds[0]) * torch.arange(dim)
        for idx in adj_idx:
            if x[:, idx] > bounds[1][idx]:
                x[:, idx] = 2. * bounds[1][idx] - x[:, idx]
            elif x[:, idx] < bounds[0][idx]:
                x[:, idx] = 2. * bounds[0][idx] - x[:, idx]
            p[:, idx] = -p[:, idx]
        flag = torch.any(torch.logical_and(x > bounds[1], x < bounds[0]))
    return x, p


def HMCwithAccept_wc(energy, x, length, steps, epsilon, burnin, n_gap, bounds):
    shape = [i if no==0 else 1 for no,i in enumerate(x.shape)]
    def grad(z):
        return torchgrad(energy(z),z,grad_outputs=z.new_ones(z.shape[0]))[0]
    torch.set_grad_enabled(False)
    E = energy(x)
    torch.set_grad_enabled(True)
    g = grad(x.requires_grad_())
    torch.set_grad_enabled(False)
    g = g.detach()
    if torch.norm(g, 2) > 0:
        g = g / torch.norm(g, 2)
    
    x_list = []
    accept_list = []
    for l in range(length):
        p = x.new_empty(size=x.size()).normal_()
        H = ((0.5*p*p).reshape(p.shape[0], -1).sum(dim=1) + E)
        xnew = x
        gnew = g
        for _ in range(steps):
            p = p - epsilon* gnew/2.
            xnew = (xnew + epsilon * p)
            torch.set_grad_enabled(True)
            gnew = grad(xnew.requires_grad_())
            torch.set_grad_enabled(False)
            xnew = xnew.detach()
            gnew = gnew.detach()
            if torch.norm(gnew, 2) > 0:
                gnew = gnew / torch.norm(gnew, 2)
    
            p = p - epsilon* gnew/2.

        xnew, p = forConstraints(xnew, p, bounds)
        Enew = energy(xnew)
        Hnew = (0.5*p*p).reshape(p.shape[0], -1).sum(dim=1) + Enew
        diff = H-Hnew
        accept = (diff.exp() >= diff.uniform_()).to(x)

        E = accept*Enew + (1.-accept)*E
        acceptMask = accept.reshape(shape)
        x = acceptMask*xnew + (1.-acceptMask)*x
        g = acceptMask*gnew + (1.-acceptMask)*g
        x_list.append(x)
        accept_list.append(accept)

    torch.set_grad_enabled(True)
    return x_list[burnin::n_gap], accept_list[burnin::n_gap]


def HMC(*args,**kwargs):
    x, _ = HMCwithAccept(*args,**kwargs)
    return unique(x)


def HMC_wc(*args, **kwargs):
    x, _ = HMCwithAccept_wc(*args, **kwargs)
    return unique(x)

class HMCsampler(nn.Module):
    def __init__(self,energy,nvars, epsilon=0.01, interSteps=10 , thermalSteps = 10):
        super(HMCsampler,self).__init__()
        self.nvars = nvars
        self.energy = energy
        self.interSteps = interSteps
        self.inital = HMC(self.energy,torch.randn(nvars),thermalSteps,interSteps)

    def step(self):
        self.inital = HMC(self.energy,self.inital,1,interSteps,epsilon)
        return self.inital
