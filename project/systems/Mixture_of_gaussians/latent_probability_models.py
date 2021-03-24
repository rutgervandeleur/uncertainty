import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal as Normal_distr
import random
from torch.nn import functional as F

class Normal(nn.Module):

    def __init__(self, parameters, latent_dim, encoder_out_dim):
        ### we need to define the device because of lightning issues
        super().__init__()

        self.device = 'cuda'
        self.p_mu, self.p_std = parameters
        self.num_params = len(parameters)
        self.latent_dim = latent_dim
        self.p = self.define_prior()
        self.fc_mu = nn.Linear(encoder_out_dim, self.latent_dim).to(self.device)
        self.fc_var = nn.Linear(encoder_out_dim, self.latent_dim).to(self.device)
        self.latent_param_network = [self.fc_mu, self.fc_var]


    def define_prior(self):
        return Normal_distr(torch.full([self.latent_dim], self.p_mu).float().to(self.device),
                                                  torch.full([self.latent_dim], self.p_std ).float().to(self.device))
    def sample_prior(self, num_samples):
        return self.p.sample_n(num_samples)


    def q(self, latent_params):
        mu, std = latent_params
        return Normal_distr(mu, std)

    def get_z(self, q):
        return q.rsample()


    def get_param_network(self, encoder_out_len):
        fc_mu = nn.Linear(encoder_out_len, self.latent_dim).to(self.device)
        fc_var = nn.Linear(encoder_out_len, self.latent_dim).to(self.device)


        return [fc_mu, fc_var]

    def get_latent_params(self, encoder_out):
        mu = self.latent_param_network[0](encoder_out)
        log_var = self.latent_param_network[1](encoder_out)
        std = torch.exp(torch.sigmoid(log_var))

        return [mu, std]

    def sample_q(self, q):
        return q.rsample()

    def KL(self, p, q, z):
        kl = torch.distributions.kl.kl_divergence(q,p)
        return kl.mean()


class Isotropic_Normal(Normal):
    def __init__(self, latent_dim, encoder_out_dim):
        super(Isotropic_Normal, self).__init__([0,1], latent_dim, encoder_out_dim)

class Isotropic_Normal_With_Mean(Normal):
    def __init__(self, mean, latent_dim, encoder_out_dim):
        super(Isotropic_Normal, self).__init__([mean,1], latent_dim, encoder_out_dim)


class Expert_Feature_Pretrained_Normal(Normal):
    ### when pretrained for the expert features, the means are just identity functions
    def __init__(self, latent_dim, encoder_out_dim):
        super(Expert_Feature_Pretrained_Normal, self).__init__([0,1], latent_dim, encoder_out_dim)

    def get_param_network(self, encoder_out_len):
        id_mu = nn.Identity().to(self.device)
        fc_var = nn.Linear(encoder_out_len, self.latent_dim).to(self.device)


        return [id_mu, fc_var]

class LogNormal(Normal):

    def __init__(self, latent_dim, encoder_out_dim):
        super(LogNormal, self).__init__([0,1], latent_dim, encoder_out_dim)

    def define_prior(self):
        return torch.distributions.LogNormal(torch.full([self.latent_dim], self.p_mu).float().to(self.device),
                                                  torch.full([self.latent_dim], self.p_std ).float().to(self.device))

    def q(self, latent_params):
        mu, std = latent_params
        return torch.distributions.LogNormal(mu, std)

class Mixture_of_Gaussians(Normal):

    ## 4 gaussians for the 4 triage cat

    def __init__(self, latent_dim, encoder_out_dim):
        super(Mixture_of_Gaussians, self).__init__([0,1], latent_dim, encoder_out_dim)

    def define_prior(self):
        return [
            Normal_distr(torch.full([self.latent_dim], -4).float().to(self.device),
                                                  torch.full([self.latent_dim], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.latent_dim], -2).float().to(self.device),
                                                  torch.full([self.latent_dim], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.latent_dim], 0).float().to(self.device),
                                                  torch.full([self.latent_dim], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.latent_dim], 2).float().to(self.device),
                                                  torch.full([self.latent_dim], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.latent_dim], 4).float().to(self.device),
                                                  torch.full([self.latent_dim], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.latent_dim], -6).float().to(self.device),
                                                  torch.full([self.latent_dim], 1 ).float().to(self.device))
        ]                                            

    def KL(self, p, q, z, triage_mask, print_acc = False):
        '''
        triage_cats = [1,2,3,4]
        mc_sample_num = 1024
        samples = q.sample([mc_sample_num])

        probs_prior = torch.stack([p[triage_cat-1].log_prob(samples) for triage_cat in triage_cats],0)
        latent_weights = q.log_prob(samples) 
        probs = torch.sum(probs_prior + latent_weights,(1,3))

        probs_std = torch.std(probs_prior + latent_weights ,(1))
        focal = probs_std 
       
        triage_labels = torch.sum(triage_mask, 2)
        triage_labels = torch.max(triage_labels, 0)[1]

        max_probs = torch.max(probs, 0)[1] 
        acc = max_probs == triage_labels

        if print_acc:
            print("train acc: ", torch.sum(acc)/acc.shape[0])
        '''
        kl_all = torch.stack([torch.distributions.kl.kl_divergence(q,triage_p) for triage_p in self.p], 0)


        #kl_all *= focal
        ##kl = kl_all * (kl_all>0.05)
     
        kl = kl_all * triage_mask
        ##kl_large_enough = kl > 0.5
        ##kl *= kl_large_enough
        return kl.mean(), kl_all ##, torch.sum(acc)/acc.shape[0]

    def KL_all_priors(self, p, q):
        kl_all = torch.stack([torch.distributions.kl.kl_divergence(q,triage_p) for triage_p in self.p], 0)
        return torch.sum(kl_all,2)

    def sample_prior(self, num_samples):
        gaussian = random.choice(self.p)
        return gaussian.sample_n(num_samples)


class Mixture_of_Gaussians_with_common_part(Normal):

    ## 4 gaussians for the 4 triage cat

    def __init__(self, latent_dim, encoder_out_dim):
        super(Mixture_of_Gaussians_with_common_part, self).__init__([0,1], latent_dim, encoder_out_dim)
        
    def define_prior(self):
        self.specific_dimensions = 10

        common_part = Normal_distr(torch.full([self.latent_dim - self.specific_dimensions], self.specific_dimensions).float().to(self.device),
                                                  torch.full([self.latent_dim - self.specific_dimensions], 1 ).float().to(self.device))

        class_specific_part = [
            Normal_distr(torch.full([self.specific_dimensions], -12).float().to(self.device),
                                                  torch.full([self.specific_dimensions], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.specific_dimensions], -8).float().to(self.device),
                                                  torch.full([self.specific_dimensions], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.specific_dimensions], -4).float().to(self.device),
                                                  torch.full([self.specific_dimensions], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.specific_dimensions], 0).float().to(self.device),
                                                  torch.full([self.specific_dimensions], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.specific_dimensions], 4).float().to(self.device),
                                                  torch.full([self.specific_dimensions], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.specific_dimensions], -14).float().to(self.device),
                                                  torch.full([self.specific_dimensions], 1 ).float().to(self.device))
        ]                 
        self.p_common = common_part
        return  class_specific_part

    def q(self, latent_params):
        mu, std = latent_params
        common_part = Normal_distr(mu[:,:-self.specific_dimensions].float().to(self.device),std[:,:-self.specific_dimensions].float().to(self.device))
        specific_part = Normal_distr(mu[:,-self.specific_dimensions:].float().to(self.device),
                                                  std[:,-self.specific_dimensions:].float().to(self.device))

        self.q_common = common_part
        return specific_part                     

    def KL(self, p, q, z, triage_mask, print_acc = False):
        '''
        triage_cats = [1,2,3,4,5,6]
        mc_sample_num = 1024
        samples = q.sample([mc_sample_num])

        probs_prior = torch.stack([p[triage_cat-1].log_prob(samples) for triage_cat in triage_cats],0)
        latent_weights = q.log_prob(samples) 
        probs = torch.sum(probs_prior + latent_weights,(1,3))
        probs = probs / torch.min(probs)
        #print(probs)
        #print(probs.shape)
        
        #probs_std = torch.std(probs_prior + latent_weights ,(1))
        #focal = 1 - full_prob_mass 
       
        triage_labels = torch.sum(triage_mask, 2)
        triage_labels = torch.max(triage_labels, 0)[1]
        #print(triage_labels.shape)

        
        max_probs = torch.max(probs, 0)[1] 
        acc = max_probs == triage_labels
        
        if print_acc:
            print("train acc: ", torch.sum(acc)/acc.shape[0])
        '''
        kl_all = torch.stack([torch.distributions.kl.kl_divergence(q,triage_p) for triage_p in self.p], 0)
        kl_common = torch.distributions.kl.kl_divergence(self.q_common,self.p_common)

        #kl_common =  F.cross_entropy(probs.T, triage_labels)

       
        ##kl = kl_all * (kl_all>0.05)
        kl = kl_all * triage_mask[:,:,-self.specific_dimensions:]

        ##kl_large_enough = kl > 0.5
        ##kl *= kl_large_enough
        #kl = torch.sum(kl, (0,2)) * focal.detach()
        return kl.mean() + kl_common.mean(), kl_all ##, torch.sum(acc)/acc.shape[0]
        
    def get_z(self, q):
        common_z = self.q_common.rsample()
        specific = q.rsample()
        return torch.cat((common_z, specific), 1)

    def KL_all_priors(self, p, q):
        kl_all = torch.stack([torch.distributions.kl.kl_divergence(q,triage_p) for triage_p in self.p], 0)
        return torch.sum(kl_all,2)

    def sample_prior(self, num_samples):
        gaussian = random.choice(self.p)
        return torch.cat((self.p_common.sample_n(num_samples),gaussian.sample_n(num_samples)), 1)

class Mixture_of_Gaussians_with_regularization(Normal):

    ## 4 gaussians for the 4 triage cat

    def __init__(self, latent_dim, encoder_out_dim):
        super(Mixture_of_Gaussians_with_regularization, self).__init__([0,1], latent_dim, encoder_out_dim)

    def define_prior(self):

        self.uniforms =  [
            Normal_distr(torch.full([self.latent_dim], -4).float().to(self.device),
                                                  torch.full([self.latent_dim], 10 ).float().to(self.device)),
            Normal_distr(torch.full([self.latent_dim], -2).float().to(self.device),
                                                  torch.full([self.latent_dim], 10 ).float().to(self.device)),
            Normal_distr(torch.full([self.latent_dim], 0).float().to(self.device),
                                                  torch.full([self.latent_dim], 10 ).float().to(self.device)),
            Normal_distr(torch.full([self.latent_dim], 2).float().to(self.device),
                                                  torch.full([self.latent_dim], 10 ).float().to(self.device))
        ]      


        return [
            Normal_distr(torch.full([self.latent_dim], -4).float().to(self.device),
                                                  torch.full([self.latent_dim], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.latent_dim], -2).float().to(self.device),
                                                  torch.full([self.latent_dim], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.latent_dim], 0).float().to(self.device),
                                                  torch.full([self.latent_dim], 1 ).float().to(self.device)),
            Normal_distr(torch.full([self.latent_dim], 2).float().to(self.device),
                                                  torch.full([self.latent_dim], 1 ).float().to(self.device))
        ]                                            

    def KL(self, p, q, z, triage_mask, print_acc = False):
        triage_cats = [0,1,2,3,4,5]

        '''
        mc_sample_num = 1024
        samples = q.sample([mc_sample_num])

        probs_prior = torch.stack([p[triage_cat-1].log_prob(samples) for triage_cat in triage_cats],0)
        latent_weights = q.log_prob(samples) 
        probs = torch.sum(probs_prior + latent_weights,(1,3))

        probs_std = torch.std(probs_prior + latent_weights ,(1))
        focal = probs_std 
       
        triage_labels = torch.sum(triage_mask, 2)
        triage_labels = torch.max(triage_labels, 0)[1]

        max_probs = torch.max(probs, 0)[1] 
        acc = max_probs == triage_labels

        if print_acc:
            print("train acc: ", torch.sum(acc)/acc.shape[0])
        '''
        kl_all = torch.stack([torch.distributions.kl.kl_divergence(q,triage_p) for triage_p in self.p], 0)
        kl_uniforms = torch.stack([torch.distributions.kl.kl_divergence(q,triage_p) for triage_p in self.uniforms], 0)

        #kl_all *= focal
        ##kl = kl_all * (kl_all>0.05)
     
        kl = kl_all * triage_mask + ((kl_uniforms * (triage_mask == 0)) / 4)
        ##kl_large_enough = kl > 0.5
        ##kl *= kl_large_enough
        return kl.mean(), kl_all ##, torch.sum(acc)/acc.shape[0]

    def KL_all_priors(self, p, q):
        kl_all = torch.stack([torch.distributions.kl.kl_divergence(q,triage_p) for triage_p in self.p], 0)
        return torch.sum(kl_all,2)

    def sample_prior(self, num_samples):
        gaussian = random.choice(self.p)
        return gaussian.sample_n(num_samples)