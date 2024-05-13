import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.autograd import Variable
import time
from utils import calc_weight
from utils import evaluate
import scipy
import scipy.io as sio
import pandas as pd


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

class encoder_DV(nn.Module):
    def __init__(self, x_dim, z_dim, dim):
        super(encoder_DV, self).__init__()

        self.f1 = nn.Linear(x_dim, 128)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)

        self.mu1 = nn.Linear(128, z_dim)
        self.log_sigma1 = nn.Linear(128, z_dim)

        self.mu2 = nn.Linear(128, dim)
        self.log_sigma2 = nn.Linear(128, dim)

    def forward(self, x):
        h = self.dropout(self.bn1(self.act(self.f1(x))))

        mu1 = self.mu1(h)
        log_sigma1 = self.log_sigma1(h).clamp(-10,10)

        mu2 = self.mu2(h)
        log_sigma2 = self.log_sigma2(h).clamp(-10,10)

        return mu1, log_sigma1, mu2, log_sigma2

class encoder_single(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(encoder_single, self).__init__()

        self.f1 = nn.Linear(x_dim, 128)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)

        self.mu1 = nn.Linear(128, z_dim)
        self.log_sigma1 = nn.Linear(128, z_dim)

    def forward(self, x):
        h = self.dropout(self.bn1(self.act(self.f1(x))))

        mu1 = self.mu1(h)
        log_sigma1 = self.log_sigma1(h).clamp(-10,10)

        return mu1, log_sigma1

class encoder_moetm(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(encoder_moetm, self).__init__()

        self.f1 = nn.Linear(x_dim, 128)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)

        self.mu = nn.Linear(128, z_dim)
        self.log_sigma = nn.Linear(128, z_dim)

    def forward(self, x):
        # [num,mod_dim]-[num, tpoic]
        h = self.dropout(self.bn1(self.act(self.f1(x))))

        mu = self.mu(h)
        log_sigma = self.log_sigma(h).clamp(-10,10)

        return mu, log_sigma

class decoder_moetm(nn.Module):
    def __init__(self, mod1_dim, mod2_dim, z_dim, emd_dim, num_batch):
        super(decoder_moetm, self).__init__()

        self.alpha_mod1 = nn.Parameter(torch.randn(mod1_dim, emd_dim))
        self.alpha_mod2 = nn.Parameter(torch.randn(mod2_dim, emd_dim))
        self.beta = nn.Parameter(torch.randn(z_dim, emd_dim))
        self.mod1_batch_bias = nn.Parameter(torch.randn(num_batch, mod1_dim))
        self.mod2_batch_bias = nn.Parameter(torch.randn(num_batch, mod2_dim))
        self.Topic_mod1 = None
        self.Topic_mod2 = None

    def forward(self, theta, batch_indices, cross_prediction = False):
        self.Topic_mod1 = torch.mm(self.alpha_mod1, self.beta.t()).t()
        self.Topic_mod2 = torch.mm(self.alpha_mod2, self.beta.t()).t()

        recon_mod1 = torch.mm(theta, self.Topic_mod1)
        recon_mod1 += self.mod1_batch_bias[batch_indices]
        if cross_prediction == False:
            recon_log_mod1 = F.log_softmax(recon_mod1, dim=-1)
        else:
            recon_log_mod1 = F.softmax(recon_mod1, dim=-1)

        recon_mod2 = torch.mm(theta, self.Topic_mod2)
        recon_mod2 += self.mod2_batch_bias[batch_indices]
        if cross_prediction == False:
            recon_log_mod2 = F.log_softmax(recon_mod2, dim=-1)
        else:
            recon_log_mod2 = F.softmax(recon_mod2, dim=-1)

        return recon_log_mod1, recon_log_mod2

class decoder_DV(nn.Module):
    def __init__(self, mod1_dim, mod2_dim, z_dim, dim, emd_dim, num_batch):
        super(decoder_DV, self).__init__()

        self.alpha_mod1 = nn.Parameter(torch.randn(mod1_dim, emd_dim))
        self.alpha_mod2 = nn.Parameter(torch.randn(mod2_dim, emd_dim))
        self.beta = nn.Parameter(torch.randn(z_dim, emd_dim))
        self.beta1 = nn.Parameter(torch.randn(dim, emd_dim))
        self.beta2 = nn.Parameter(torch.randn(dim, emd_dim))
        self.mod1_batch_bias = nn.Parameter(torch.randn(num_batch, mod1_dim))
        self.mod2_batch_bias = nn.Parameter(torch.randn(num_batch, mod2_dim))
        self.Topic_mod1 = None
        self.Topic_mod2 = None

    def forward(self, theta, theta1, theta2, batch_indices, cross_prediction = False):
        self.Topic_mod1 = torch.mm(self.alpha_mod1, self.beta.t()).t() #[zdim,mod1dim]
        self.Topic_mod2 = torch.mm(self.alpha_mod2, self.beta.t()).t()

        M_1 = torch.mm(theta1, self.beta1)
        M_2 = torch.mm(theta2, self.beta2)
        M_share = torch.mm(theta, self.beta)
        M1 = torch.add(M_1, M_share)
        M2 = torch.add(M_2, M_share)  

        recon_mod1 = torch.mm(M1, self.alpha_mod1.t())
        recon_mod1 += self.mod1_batch_bias[batch_indices]
        if cross_prediction == False:
            recon_log_mod1 = F.log_softmax(recon_mod1, dim=-1)
        else:
            recon_log_mod1 = F.softmax(recon_mod1, dim=-1)

        recon_mod2 = torch.mm(M2, self.alpha_mod2.t())
        recon_mod2 += self.mod2_batch_bias[batch_indices]
        if cross_prediction == False:
            recon_log_mod2 = F.log_softmax(recon_mod2, dim=-1)
        else:
            recon_log_mod2 = F.softmax(recon_mod2, dim=-1)

        return recon_log_mod1, recon_log_mod2

class decoder_single(nn.Module):
    def __init__(self, mod1_dim, z_dim, dim, emd_dim, num_batch):
        super(decoder_single, self).__init__()

        self.alpha_mod1 = nn.Parameter(torch.randn(mod1_dim, emd_dim))
        self.beta = nn.Parameter(torch.randn(z_dim, emd_dim))
        self.beta1 = nn.Parameter(torch.randn(dim, emd_dim))
        self.mod1_batch_bias = nn.Parameter(torch.randn(num_batch, mod1_dim))
        self.Topic_mod1 = None


    def forward(self, theta, theta1, batch_indices, cross_prediction = False):
        self.Topic_mod1 = torch.mm(self.alpha_mod1, self.beta.t()).t() #[zdim,mod1dim]

        M_1 = torch.mm(theta1, self.beta1)
        M_share = torch.mm(theta, self.beta)
        M1 = torch.add(M_1, M_share) 

        recon_mod1 = torch.mm(M1, self.alpha_mod1.t())
        recon_mod1 += self.mod1_batch_bias[batch_indices]
        if cross_prediction == False:
            recon_log_mod1 = F.log_softmax(recon_mod1, dim=-1)
        else:
            recon_log_mod1 = F.softmax(recon_mod1, dim=-1)

        return recon_log_mod1
    
class decoder_DV_nolangda(nn.Module):
    def __init__(self, mod1_dim, mod2_dim, z_dim, dim, emd_dim, num_batch):
        super(decoder_DV_nolangda, self).__init__()

        self.alpha_mod1 = nn.Parameter(torch.randn(mod1_dim, emd_dim))
        self.alpha_mod2 = nn.Parameter(torch.randn(mod2_dim, emd_dim))
        self.beta = nn.Parameter(torch.randn(z_dim, emd_dim))
        self.beta1 = nn.Parameter(torch.randn(dim, emd_dim))
        self.beta2 = nn.Parameter(torch.randn(dim, emd_dim))
        #self.mod1_batch_bias = nn.Parameter(torch.randn(num_batch, mod1_dim))
        #self.mod2_batch_bias = nn.Parameter(torch.randn(num_batch, mod2_dim))
        self.Topic_mod1 = None
        self.Topic_mod2 = None

    def forward(self, theta, theta1, theta2, batch_indices, cross_prediction = False):
        self.Topic_mod1 = torch.mm(self.alpha_mod1, self.beta.t()).t() #[zdim,mod1dim]
        self.Topic_mod2 = torch.mm(self.alpha_mod2, self.beta.t()).t()

        M_1 = torch.mm(theta1, self.beta1)
        M_2 = torch.mm(theta2, self.beta2)
        M_share = torch.mm(theta, self.beta)
        M1 = torch.add(M_1, M_share)
        M2 = torch.add(M_2, M_share)  

        recon_mod1 = torch.mm(M1, self.alpha_mod1.t())
        #recon_mod1 += self.mod1_batch_bias[batch_indices]
        if cross_prediction == False:
            recon_log_mod1 = F.log_softmax(recon_mod1, dim=-1)
        else:
            recon_log_mod1 = F.softmax(recon_mod1, dim=-1)

        recon_mod2 = torch.mm(M2, self.alpha_mod2.t())
        #recon_mod2 += self.mod2_batch_bias[batch_indices]
        if cross_prediction == False:
            recon_log_mod2 = F.log_softmax(recon_mod2, dim=-1)
        else:
            recon_log_mod2 = F.softmax(recon_mod2, dim=-1)

        return recon_log_mod1, recon_log_mod2

class decoder_scMM(nn.Module):
    def __init__(self, mod1_dim, mod2_dim, z_dim, num_batch):
        super(decoder_scMM, self).__init__()

        self.f1 = nn.Linear(z_dim, 128)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)
        self.recon1 = nn.Linear(128, mod1_dim)

        self.f2 = nn.Linear(z_dim, 128)
        self.act = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)
        self.recon2 = nn.Linear(128, mod2_dim)

    def forward(self, theta, batch_indices, cross_prediction = False):
        recon_mod1 = self.recon1(self.dropout(self.bn1(self.act(self.f1(theta)))))
        if cross_prediction == False:
            recon_log_mod1 = F.log_softmax(recon_mod1, dim=-1)
        else:
            recon_log_mod1 = F.softmax(recon_mod1, dim=-1)

        recon_mod2 = self.recon2(self.dropout(self.bn2(self.act(self.f2(theta)))))
        if cross_prediction == False:
            recon_log_mod2 = F.log_softmax(recon_mod2, dim=-1)
        else:
            recon_log_mod2 = F.softmax(recon_mod2, dim=-1)

        return recon_log_mod1, recon_log_mod2

class decoder_multigrate(nn.Module):
    def __init__(self, mod1_dim, mod2_dim, z_dim, num_batch):
        super(decoder_multigrate, self).__init__()

        self.f1 = nn.Linear(z_dim, 128)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)
        self.recon1 = nn.Linear(128, mod1_dim)
        self.recon2 = nn.Linear(128, mod2_dim)

    def forward(self, theta, batch_indices, cross_prediction = False):
        the = self.dropout(self.bn1(self.act(self.f1(theta))))

        recon_mod1 = self.recon1(the)
        if cross_prediction == False:
            recon_log_mod1 = F.log_softmax(recon_mod1, dim=-1)
        else:
            recon_log_mod1 = F.softmax(recon_mod1, dim=-1)

        recon_mod2 = self.recon2(the)
        if cross_prediction == False:
            recon_log_mod2 = F.log_softmax(recon_mod2, dim=-1)
        else:
            recon_log_mod2 = F.softmax(recon_mod2, dim=-1)

        return recon_log_mod1, recon_log_mod2

class decoder_cobolt(nn.Module):
    def __init__(self, mod1_dim, mod2_dim, z_dim, dim, num_batch):
        super(decoder_cobolt, self).__init__()

        self.f1 = nn.Linear(z_dim+dim, 128)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)
        self.recon1 = nn.Linear(128, mod1_dim)

        self.f2 = nn.Linear(z_dim+dim, 128)
        self.act = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)
        self.recon2 = nn.Linear(128, mod2_dim)

    def forward(self, theta1, theta2, batch_indices, cross_prediction = False):
        recon_mod1 = self.recon1(self.dropout(self.bn1(self.act(self.f1(theta1)))))
        if cross_prediction == False:
            recon_log_mod1 = F.log_softmax(recon_mod1, dim=-1)
        else:
            recon_log_mod1 = F.softmax(recon_mod1, dim=-1)

        recon_mod2 = self.recon2(self.dropout(self.bn2(self.act(self.f2(theta2)))))
        if cross_prediction == False:
            recon_log_mod2 = F.log_softmax(recon_mod2, dim=-1)
        else:
            recon_log_mod2 = F.softmax(recon_mod2, dim=-1)

        return recon_log_mod1, recon_log_mod2

def build_cobolt(input_dim_mod1, input_dim_mod2, num_batch, num_topic=50, num_indep = 10, emd_dim = 400):

    encoder_mod1 = encoder_DV(x_dim=input_dim_mod1, z_dim=num_topic, dim = num_indep).cuda()
    encoder_mod2 = encoder_DV(x_dim=input_dim_mod2, z_dim=num_topic, dim = num_indep).cuda()
    decoder_all = decoder_cobolt(mod1_dim=input_dim_mod1, mod2_dim=input_dim_mod2, z_dim=num_topic, dim = num_indep, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': encoder_mod2.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, encoder_mod2, decoder_all, optimizer

def build_scMM(input_dim_mod1, input_dim_mod2, num_batch, num_topic=50, num_indep = 10, emd_dim = 400):

    encoder_mod1 = encoder_single(x_dim=input_dim_mod1, z_dim=num_topic).cuda()
    encoder_mod2 = encoder_single(x_dim=input_dim_mod2, z_dim=num_topic).cuda()
    decoder_all = decoder_scMM(mod1_dim=input_dim_mod1, mod2_dim=input_dim_mod2, z_dim=num_topic, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': encoder_mod2.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, encoder_mod2, decoder_all, optimizer

def build_multigrate(input_dim_mod1, input_dim_mod2, num_batch, num_topic=50, num_indep = 10, emd_dim = 400):

    encoder_mod1 = encoder_single(x_dim=input_dim_mod1, z_dim=num_topic).cuda()
    encoder_mod2 = encoder_single(x_dim=input_dim_mod2, z_dim=num_topic).cuda()
    decoder_all = decoder_multigrate(mod1_dim=input_dim_mod1, mod2_dim=input_dim_mod2, z_dim=num_topic, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': encoder_mod2.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, encoder_mod2, decoder_all, optimizer

def build_moDVTM(input_dim_mod1, input_dim_mod2, num_batch, num_topic=50, num_indep = 10, emd_dim=400):
    encoder_mod1 = encoder_DV(x_dim=input_dim_mod1, z_dim=num_topic, dim = num_indep).cuda()
    encoder_mod2 = encoder_DV(x_dim=input_dim_mod2, z_dim=num_topic, dim = num_indep).cuda()
    decoder_all = decoder_DV(mod1_dim=input_dim_mod1, mod2_dim=input_dim_mod2, z_dim=num_topic, dim = num_indep, emd_dim=emd_dim, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': encoder_mod2.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, encoder_mod2, decoder_all, optimizer

def build_moDVTM_rna(input_dim_mod1, num_batch, num_topic=50, num_indep = 10, emd_dim=400):
    encoder_mod1 = encoder_DV(x_dim=input_dim_mod1, z_dim=num_topic, dim = num_indep).cuda()
    #encoder_mod2 = encoder_single(x_dim=input_dim_mod2, z_dim=num_topic).cuda()
    decoder_all = decoder_single(mod1_dim=input_dim_mod1, z_dim=num_topic, dim = num_indep, emd_dim=emd_dim, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, decoder_all, optimizer

def build_moDVTM_dna(input_dim_mod1, num_batch, num_topic=50, num_indep = 10, emd_dim=400):
    encoder_mod1 = encoder_DV(x_dim=input_dim_mod1, z_dim=num_topic, dim = num_indep).cuda()
    #encoder_mod2 = encoder_single(x_dim=input_dim_mod2, z_dim=num_topic).cuda()
    decoder_all = decoder_single(mod1_dim=input_dim_mod1, z_dim=num_topic, dim = num_indep, emd_dim=emd_dim, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, decoder_all, optimizer

def build_moDVTM_protein(input_dim_mod1, num_batch, num_topic=50, num_indep = 10, emd_dim=400):
    encoder_mod1 = encoder_DV(x_dim=input_dim_mod1, z_dim=num_topic, dim = num_indep).cuda()
    #encoder_mod2 = encoder_single(x_dim=input_dim_mod2, z_dim=num_topic).cuda()
    decoder_all = decoder_single(mod1_dim=input_dim_mod1, z_dim=num_topic, dim = num_indep, emd_dim=emd_dim, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, decoder_all, optimizer

def build_moDVTM_nolangda(input_dim_mod1, input_dim_mod2, num_batch, num_topic=50, num_indep = 10, emd_dim=400):
    encoder_mod1 = encoder_DV(x_dim=input_dim_mod1, z_dim=num_topic, dim = num_indep).cuda()
    encoder_mod2 = encoder_DV(x_dim=input_dim_mod2, z_dim=num_topic, dim = num_indep).cuda()
    decoder_all = decoder_DV_nolangda(mod1_dim=input_dim_mod1, mod2_dim=input_dim_mod2, z_dim=num_topic, dim = num_indep, emd_dim=emd_dim, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': encoder_mod2.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, encoder_mod2, decoder_all, optimizer

def build_moDVTM_moe(input_dim_mod1, input_dim_mod2, num_batch, num_topic=50, num_indep = 10, emd_dim=400):
    encoder_mod1 = encoder_DV(x_dim=input_dim_mod1, z_dim=num_topic, dim = num_indep).cuda()
    encoder_mod2 = encoder_DV(x_dim=input_dim_mod2, z_dim=num_topic, dim = num_indep).cuda()
    decoder_all = decoder_DV(mod1_dim=input_dim_mod1, mod2_dim=input_dim_mod2, z_dim=num_topic, dim = num_indep, emd_dim=emd_dim, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': encoder_mod2.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, encoder_mod2, decoder_all, optimizer

def build_moDVTM_share(input_dim_mod1, input_dim_mod2, num_batch, num_topic=50,  num_indep = 10, emd_dim=400):

    encoder_mod1 = encoder_moetm(x_dim=input_dim_mod1, z_dim=num_topic).cuda()
    encoder_mod2 = encoder_moetm(x_dim=input_dim_mod2, z_dim=num_topic).cuda()
    decoder_all = decoder_moetm(mod1_dim=input_dim_mod1, mod2_dim=input_dim_mod2, z_dim=num_topic, emd_dim=emd_dim, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': encoder_mod2.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, encoder_mod2, decoder_all, optimizer

class Trainer_moDVTM_inte(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
        mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)
        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod3.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod3.unsqueeze(0)), dim=0)
        mu_share, log_sigma_share = self.experts(Mu, Log_sigma)
        Theta_share = self.reparameterize(mu_share, log_sigma_share) #log-normal distribution

        Theta_st = self.reparameterize(mu_mod2, log_sigma_mod2) #log-normal distribution
        Theta_rd = self.reparameterize(mu_mod4, log_sigma_mod4)#log-normal distribution

        dim_share = Theta_share.shape[1]
        dim_st = Theta_st.shape[1]
        dim_rd = Theta_rd.shape[1]

        Theta_com = torch.cat((Theta_share, Theta_st, Theta_rd), dim=1)
        Theta = F.softmax(Theta_com, dim=-1)

        Theta_share = Theta[:, :dim_share]
        Theta_st = Theta[:, dim_share:dim_share+dim_st]
        Theta_rd = Theta[:, dim_share+dim_st:]

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta_share, Theta_st, Theta_rd, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu_share, log_sigma_share).mean()
        # KL_st = self.get_kl(mu_mod2, log_sigma_mod2).mean()
        # KL_rd = self.get_kl(mu_mod4, log_sigma_mod4).mean()
        # KL = KL_share + KL_st + KL_rd
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
            mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)
            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod3.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod3.unsqueeze(0)), dim=0)
            mu_share, log_sigma_share = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def get_embed_best(self, x_mod1, x_mod2):
        self.best_encoder_mod1.eval()
        self.best_encoder_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
            mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)
            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod3.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod3.unsqueeze(0)), dim=0)
            mu_share, log_sigma_share = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_moDVTM_moe_inte(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
        mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

        # mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)
        # Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod3.unsqueeze(0)), dim=0)
        # Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod3.unsqueeze(0)), dim=0)
        # mu_share, log_sigma_share = self.experts(Mu, Log_sigma)
        # Theta_share = self.reparameterize(mu_share, log_sigma_share) #log-normal distribution

        mu_share = 0.5*(mu_mod1 + mu_mod3)
        sigma_mod1 = torch.exp(log_sigma_mod1 * 2)
        sigma_mod3 = torch.exp(log_sigma_mod3 * 2)
        avg_sigma = (sigma_mod1 + sigma_mod3) / 4
        log_sigma_share = torch.log(avg_sigma) / 2
        Theta_share = self.reparameterize(mu_share, log_sigma_share)

        Theta_st = self.reparameterize(mu_mod2, log_sigma_mod2) #log-normal distribution
        Theta_rd = self.reparameterize(mu_mod4, log_sigma_mod4)#log-normal distribution

        dim_share = Theta_share.shape[1]
        dim_st = Theta_st.shape[1]
        dim_rd = Theta_rd.shape[1]

        Theta_com = torch.cat((Theta_share, Theta_st, Theta_rd), dim=1)
        Theta = F.softmax(Theta_com, dim=-1)

        Theta_share = Theta[:, :dim_share]
        Theta_st = Theta[:, dim_share:dim_share+dim_st]
        Theta_rd = Theta[:, dim_share+dim_st:]

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta_share, Theta_st, Theta_rd, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu_share, log_sigma_share).mean()
        # KL_st = self.get_kl(mu_mod2, log_sigma_mod2).mean()
        # KL_rd = self.get_kl(mu_mod4, log_sigma_mod4).mean()
        # KL = KL_share + KL_st + KL_rd
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
            mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

            mu_share = 0.5*(mu_mod1 + mu_mod3)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def get_embed_best(self, x_mod1, x_mod2):
        self.best_encoder_mod1.eval()
        self.best_encoder_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
            mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

            mu_share = 0.5*(mu_mod1 + mu_mod3)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_cobolt_inte(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
        mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)
        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod3.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod3.unsqueeze(0)), dim=0)
        mu_share, log_sigma_share = self.experts(Mu, Log_sigma)
        Theta_share = self.reparameterize(mu_share, log_sigma_share) #log-normal distribution

        Theta_st = self.reparameterize(mu_mod2, log_sigma_mod2) #log-normal distribution
        Theta_rd = self.reparameterize(mu_mod4, log_sigma_mod4)#log-normal distribution

        # dim_share = Theta_share.shape[1]
        # dim_st = Theta_st.shape[1]
        # dim_rd = Theta_rd.shape[1]

        # Theta_com = torch.cat((Theta_share, Theta_st, Theta_rd), dim=1)
        # Theta = F.softmax(Theta_com, dim=-1)

        # Theta_share = Theta[:, :dim_share]
        # Theta_st = Theta[:, dim_share:dim_share+dim_st]
        # Theta_rd = Theta[:, dim_share+dim_st:]
        Theta_1 = torch.cat((Theta_share, Theta_st), dim=1)
        Theta_2 = torch.cat((Theta_share, Theta_rd), dim=1)

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta_1, Theta_2, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu_share, log_sigma_share).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
            mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)
            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod3.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod3.unsqueeze(0)), dim=0)
            mu_share, log_sigma_share = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def get_embed_best(self, x_mod1, x_mod2):
        self.best_encoder_mod1.eval()
        self.best_encoder_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
            mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)
            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod3.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod3.unsqueeze(0)), dim=0)
            mu_share, log_sigma_share = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_moDVTM_nolangda_inte(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
        mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)
        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod3.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod3.unsqueeze(0)), dim=0)
        mu_share, log_sigma_share = self.experts(Mu, Log_sigma)
        Theta_share = self.reparameterize(mu_share, log_sigma_share) #log-normal distribution

        Theta_st = self.reparameterize(mu_mod2, log_sigma_mod2) #log-normal distribution
        Theta_rd = self.reparameterize(mu_mod4, log_sigma_mod4)#log-normal distribution

        dim_share = Theta_share.shape[1]
        dim_st = Theta_st.shape[1]
        dim_rd = Theta_rd.shape[1]

        Theta_com = torch.cat((Theta_share, Theta_st, Theta_rd), dim=1)
        Theta = F.softmax(Theta_com, dim=-1)

        Theta_share = Theta[:, :dim_share]
        Theta_st = Theta[:, dim_share:dim_share+dim_st]
        Theta_rd = Theta[:, dim_share+dim_st:]

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta_share, Theta_st, Theta_rd, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu_share, log_sigma_share).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
            mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)
            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod3.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod3.unsqueeze(0)), dim=0)
            mu_share, log_sigma_share = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def get_embed_best(self, x_mod1, x_mod2):
        self.best_encoder_mod1.eval()
        self.best_encoder_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)
            mu_mod3, log_sigma_mod3, mu_mod4, log_sigma_mod4= self.encoder_mod2(x_mod2)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)
            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod3.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod3.unsqueeze(0)), dim=0)
            mu_share, log_sigma_share = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma
    
class Trainer_multigrate_inte(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1= self.encoder_mod1(x_mod1)
        mu_mod2, log_sigma_mod2= self.encoder_mod2(x_mod2)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2= self.encoder_mod2(x_mod2)

            mu_share = 0.5*(mu_mod1 + mu_mod2)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def get_embed_best(self, x_mod1, x_mod2):
        self.best_encoder_mod1.eval()
        self.best_encoder_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2= self.encoder_mod2(x_mod2)

            mu_share = 0.5*(mu_mod1 + mu_mod2)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_scMM_inte(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1= self.encoder_mod1(x_mod1)
        mu_mod2, log_sigma_mod2= self.encoder_mod2(x_mod2)

        # mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)
        # Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod3.unsqueeze(0)), dim=0)
        # Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod3.unsqueeze(0)), dim=0)
        # mu_share, log_sigma_share = self.experts(Mu, Log_sigma)
        mu = 0.5*(mu_mod1 + mu_mod2)
        sigma_mod1 = torch.exp(log_sigma_mod1 * 2)
        sigma_mod2 = torch.exp(log_sigma_mod2 * 2)
        avg_sigma = (sigma_mod1 + sigma_mod2) / 4
        log_sigma = torch.log(avg_sigma) / 2

        # Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution
        Theta = self.reparameterize(mu, log_sigma) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2= self.encoder_mod2(x_mod2)

            mu_share = 0.5*(mu_mod1 + mu_mod2)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def get_embed_best(self, x_mod1, x_mod2):
        self.best_encoder_mod1.eval()
        self.best_encoder_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2= self.encoder_mod2(x_mod2)

            mu_share = 0.5*(mu_mod1 + mu_mod2)

        out = {}
        out['delta'] = np.array(mu_share)
        return out

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_moDVTM_share_inte(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
        mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False) # 创建[1,num,topic]的零向量

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0) #[1,num,topic]-[3,num,topic]
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0) #[1,num,topic]-[3,num,topic]

            mu, log_sigma = self.experts(Mu, Log_sigma) # [num,topic]

        out = {}
        out['delta'] = np.array(mu)
        return out

    def get_embed_best(self, x_mod1, x_mod2):
        self.best_encoder_mod1.eval()
        self.best_encoder_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.best_encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.best_encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu)
        return out

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_moDVTM_rna_inte(object):
    def __init__(self, encoder_mod1, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None

    def train(self, x_mod1, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)
        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0)), dim=0)
        mu_share, log_sigma_share = self.experts(Mu, Log_sigma)
        Theta_share = self.reparameterize(mu_share, log_sigma_share) #log-normal distribution

        Theta_st = self.reparameterize(mu_mod2, log_sigma_mod2) #log-normal distribution

        dim_share = Theta_share.shape[1]
        dim_st = Theta_st.shape[1]

        Theta_com = torch.cat((Theta_share, Theta_st), dim=1)
        Theta = F.softmax(Theta_com, dim=-1)

        Theta_share = Theta[:, :dim_share]
        Theta_st = Theta[:, dim_share:]

        recon_log_mod1 = self.decoder(Theta_share, Theta_st, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()

        KL = self.get_kl(mu_mod1, log_sigma_mod1).mean()
        Loss = nll_mod1 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1):
        self.encoder_mod1.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)

        out = {}
        out['delta'] = np.array(mu_mod1)
        return out

    def get_embed_best(self, x_mod1):
        self.best_encoder_mod1.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)

        out = {}
        out['delta'] = np.array(mu_mod1)
        return out

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_moDVTM_dna_inte(object):
    def __init__(self, encoder_mod1, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None

    def train(self, x_mod1, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)
        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0)), dim=0)
        mu_share, log_sigma_share = self.experts(Mu, Log_sigma)
        Theta_share = self.reparameterize(mu_share, log_sigma_share) #log-normal distribution

        Theta_st = self.reparameterize(mu_mod2, log_sigma_mod2) #log-normal distribution

        dim_share = Theta_share.shape[1]
        dim_st = Theta_st.shape[1]

        Theta_com = torch.cat((Theta_share, Theta_st), dim=1)
        Theta = F.softmax(Theta_com, dim=-1)

        Theta_share = Theta[:, :dim_share]
        Theta_st = Theta[:, dim_share:]

        recon_log_mod1 = self.decoder(Theta_share, Theta_st, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()

        KL = self.get_kl(mu_mod1, log_sigma_mod1).mean()
        Loss = nll_mod1 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1):
        self.encoder_mod1.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)

        out = {}
        out['delta'] = np.array(mu_mod1)
        return out

    def get_embed_best(self, x_mod1):
        self.best_encoder_mod1.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)

        out = {}
        out['delta'] = np.array(mu_mod1)
        return out

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_moDVTM_protein_inte(object):
    def __init__(self, encoder_mod1, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None

    def train(self, x_mod1, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)
        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0)), dim=0)
        mu_share, log_sigma_share = self.experts(Mu, Log_sigma)
        Theta_share = self.reparameterize(mu_share, log_sigma_share) #log-normal distribution

        Theta_st = self.reparameterize(mu_mod2, log_sigma_mod2) #log-normal distribution

        dim_share = Theta_share.shape[1]
        dim_st = Theta_st.shape[1]

        Theta_com = torch.cat((Theta_share, Theta_st), dim=1)
        Theta = F.softmax(Theta_com, dim=-1)

        Theta_share = Theta[:, :dim_share]
        Theta_st = Theta[:, dim_share:]

        recon_log_mod1 = self.decoder(Theta_share, Theta_st, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()

        KL = self.get_kl(mu_mod1, log_sigma_mod1).mean()
        Loss = nll_mod1 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1):
        self.encoder_mod1.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)

        out = {}
        out['delta'] = np.array(mu_mod1)
        return out

    def get_embed_best(self, x_mod1):
        self.best_encoder_mod1.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1, mu_mod2, log_sigma_mod2 = self.encoder_mod1(x_mod1)

        out = {}
        out['delta'] = np.array(mu_mod1)
        return out

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_moDVTM_share_impu_another2rna(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod, log_sigma_mod = self.encoder_mod2(x_mod2)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod.shape[1]), use_cuda=True)


        Mu = torch.cat((mu_prior, mu_mod.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reconstruction(self, x_mod1, x_mod2, batch_indices):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)

            # mu_zero = torch.zeros(mu_mod1.shape)
            # log_sigma_one = torch.ones(log_sigma_mod2.shape)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            _, recon_mod2 = self.decoder(Theta, batch_indices, cross_prediction=True)

            Mu = torch.cat((mu_prior, mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod2.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution
            recon_mod1, _ = self.decoder(Theta, batch_indices, cross_prediction=True)

            return recon_mod1, recon_mod2

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_multigrate_impu_another2rna(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod, log_sigma_mod = self.encoder_mod2(x_mod2)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod.shape[1]), use_cuda=True)


        Mu = torch.cat((mu_prior, mu_mod.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reconstruction(self, x_mod1, x_mod2, batch_indices):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)

            # mu_zero = torch.zeros(mu_mod1.shape)
            # log_sigma_one = torch.ones(log_sigma_mod2.shape)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            _, recon_mod2 = self.decoder(Theta, batch_indices, cross_prediction=True)

            Mu = torch.cat((mu_prior, mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod2.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution
            recon_mod1, _ = self.decoder(Theta, batch_indices, cross_prediction=True)

            return recon_mod1, recon_mod2

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_scMM_impu_another2rna(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod, log_sigma_mod = self.encoder_mod2(x_mod2)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod.shape[1]), use_cuda=True)


        Mu = torch.cat((mu_prior, mu_mod.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reconstruction(self, x_mod1, x_mod2, batch_indices):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)

            # mu_zero = torch.zeros(mu_mod1.shape)
            # log_sigma_one = torch.ones(log_sigma_mod2.shape)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            _, recon_mod2 = self.decoder(Theta, batch_indices, cross_prediction=True)

            Mu = torch.cat((mu_prior, mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod2.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution
            recon_mod1, _ = self.decoder(Theta, batch_indices, cross_prediction=True)

            return recon_mod1, recon_mod2

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_moDVTM_share_impu_rna2another(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod, log_sigma_mod = self.encoder_mod1(x_mod1)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod.shape[1]), use_cuda=True)


        Mu = torch.cat((mu_prior, mu_mod.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reconstruction(self, x_mod1, x_mod2, batch_indices):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)

            # mu_zero = torch.zeros(mu_mod1.shape)
            # log_sigma_one = torch.ones(log_sigma_mod2.shape)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            _, recon_mod2 = self.decoder(Theta, batch_indices, cross_prediction=True)

            Mu = torch.cat((mu_prior, mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod2.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution
            recon_mod1, _ = self.decoder(Theta, batch_indices, cross_prediction=True)

            return recon_mod1, recon_mod2

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_multigrate_impu_rna2another(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod, log_sigma_mod = self.encoder_mod1(x_mod1)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod.shape[1]), use_cuda=True)


        Mu = torch.cat((mu_prior, mu_mod.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reconstruction(self, x_mod1, x_mod2, batch_indices):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)

            # mu_zero = torch.zeros(mu_mod1.shape)
            # log_sigma_one = torch.ones(log_sigma_mod2.shape)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            _, recon_mod2 = self.decoder(Theta, batch_indices, cross_prediction=True)

            Mu = torch.cat((mu_prior, mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod2.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution
            recon_mod1, _ = self.decoder(Theta, batch_indices, cross_prediction=True)

            return recon_mod1, recon_mod2

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_scMM_impu_rna2another(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod, log_sigma_mod = self.encoder_mod1(x_mod1)

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod.shape[1]), use_cuda=True)


        Mu = torch.cat((mu_prior, mu_mod.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reconstruction(self, x_mod1, x_mod2, batch_indices):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)

            # mu_zero = torch.zeros(mu_mod1.shape)
            # log_sigma_one = torch.ones(log_sigma_mod2.shape)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            _, recon_mod2 = self.decoder(Theta, batch_indices, cross_prediction=True)

            Mu = torch.cat((mu_prior, mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod2.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution
            recon_mod1, _ = self.decoder(Theta, batch_indices, cross_prediction=True)

            return recon_mod1, recon_mod2

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma
