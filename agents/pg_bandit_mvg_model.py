import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, LogNormal
import numpy as np

from mi_classifier import MIClassifier

def get_mean_std_multiple_ps(all_means, all_var):
    sample_mean = np.mean(all_means, axis = 0)
    sample_std = np.sqrt( ( np.sum(all_var, axis = 0) + np.sum( (all_means-sample_mean)**2, axis = 0) )/len(all_means) )
    return sample_mean, sample_std

class PGBanditMVGModel(nn.Module):
    """Principal policy. Uses policy gradient."""
    def __init__(self, n_a_outputs, mi_lambda = 0.0, initialize = False, entropy_reg = 0, d = 0.0):
        super().__init__()
        
        self.n_a_outputs = n_a_outputs
        
        if initialize:
            self.logmean_params = nn.Parameter(torch.cat([torch.FloatTensor(np.arange(-2.0, 1.0, 3.0/n_a_outputs)), d* torch.ones(n_a_outputs)]).float().view(1, 2*n_a_outputs))
        else:
            self.logmean_params = nn.Parameter(d* torch.ones(2*n_a_outputs).float().view(1, 2*n_a_outputs))
        self.logstd_params = nn.Parameter(d* torch.ones(2*n_a_outputs).float().view(1, 2*n_a_outputs))
        
        self.entropy_reg = entropy_reg
        self.m_entropy = None

    def set_entropy_reg(self, er):
        self.entropy_reg = er

    def forward(self):
        logmean = self.logmean_params
        stds = torch.exp(self.logstd_params)
        return logmean, stds

    def predict(self, n):
        logmean, stds = self.forward()
        
        #mean model sample
#         cov = torch.diag_embed(stds**2) #assume no covariance
        all_mu = logmean.repeat((n, 1))
        all_std = stds.repeat((n, 1))
#         m_mean = MultivariateNormal(logmean, stds)
        
        m_model = Normal(all_mu, all_std)
        
        return m_model
         
    def sample(self, n):
        m_model= self.predict(n)
        self.m_entropy = m_model.entropy()
        log_params = m_model.sample()
        schedule_log_probs = torch.sum(m_model.log_prob(log_params), dim = 1)
        
        schedule_logmeans = log_params[:, 0:self.n_a_outputs]
        schedule_logstds = log_params[:, self.n_a_outputs:]

        return torch.exp(schedule_logmeans), torch.exp(schedule_logstds), schedule_log_probs
    
    def get_entropy(self):
        return self.m_entropy
    
    def sample_and_entropy(self, n):
        schedule_means, schedule_stds, schedule_log_probs = self.sample(n)
        m_entropy = self.get_entropy()
        return schedule_means, schedule_stds, schedule_log_probs, m_entropy
    
    def get_mean_mean_std(self):
        m_params = self.predict(1).loc[0].detach()
    
        mean_mean = torch.exp(m_params[0:self.n_a_outputs]).numpy()
        mean_std = torch.exp(m_params[self.n_a_outputs:]).numpy()
        return mean_mean, mean_std
        
        
    def get_loss(self, r, schedule_log_probs, m_entropy):
        loss = -schedule_log_probs * r - self.entropy_reg * m_entropy.mean()
        return loss.mean()

class PGBanditLNModel(nn.Module):
    """Principal policy. Uses policy gradient. Lognormal distribution"""
    def __init__(self, n_a_outputs, initialize = False, entropy_reg = 0, d = 0.5):
        super().__init__()
        
        self.n_a_outputs = n_a_outputs
        # let Z be normal(mean, std)
        # Then X is lognormal: e^(Z)
        
        #mean params are the mean and std of the log normal
        if initialize:
            
            self.mean_params = nn.Parameter(torch.cat([torch.FloatTensor(np.arange(-2.0, 1.0, 3.0/n_a_outputs)), d* torch.ones(n_a_outputs)]).float().view(1, 2*n_a_outputs))
        else:
            self.mean_params = nn.Parameter(d* torch.ones(2*n_a_outputs).float().view(1, 2*n_a_outputs))
            
        self.logstd_params = nn.Parameter(d* torch.ones(2*n_a_outputs).float().view(1, 2*n_a_outputs))
        
        self.entropy_reg = entropy_reg
        self.m_entropy = None

    def set_entropy_reg(self, er):
        self.entropy_reg = er

    def forward(self):
        mean = self.mean_params
        stds = torch.exp(self.logstd_params)
        
        return mean, stds

    def predict(self, n):
        mean, stds = self.forward()
        
        #mean model sample
        all_mu = mean.repeat((n, 1))
        all_std = stds.repeat((n, 1))
        
        m_model = LogNormal(all_mu, all_std)
        
        return m_model
    
    def get_mean_mean_std(self):
        means, stds, log_probs = self.sample(100)
        means = means.detach().numpy()
        stds = stds.detach().numpy()
        mean_mean, mean_std = get_mean_std_multiple_ps(means, stds**2)

#         m_params = self.predict(1).loc[0].detach()
    
#         mean_log = m_params[0:self.n_a_outputs]
#         std_log = m_params[self.n_a_outputs:]
        
#         mean_mean = torch.exp(mean_log + (std_log**2)/2.0)
#         mean_std = (torch.exp(std_log**2)-1)*torch.exp(2*mean_log + std_log**2)
        
        return mean_mean, mean_std
         
    def sample(self, n):
        m_model= self.predict(n)
        self.m_entropy = m_model.entropy()
        params = m_model.sample()
        schedule_log_probs = torch.sum(m_model.log_prob(params), dim = 1)
        
        schedule_means = params[:, 0:self.n_a_outputs]
        schedule_stds = params[:, self.n_a_outputs:]

        return schedule_means, schedule_stds, schedule_log_probs
    
    def get_entropy(self):
        return self.m_entropy
    
    def sample_and_entropy(self, n):
        schedule_means, schedule_stds, schedule_log_probs = self.sample(n)
        m_entropy = self.get_entropy()
        return schedule_means, schedule_stds, schedule_log_probs, m_entropy
        
        
    def get_loss(self, r, schedule_log_probs, m_entropy):
        loss = -schedule_log_probs * r - self.entropy_reg * m_entropy.mean()
        return loss.mean()