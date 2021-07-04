import torch
from torch import nn


class TaskLoss(nn.Module):
    
    def __init__(self, lamb=1):
        super(TaskLoss, self).__init__()

        self.lamb = lamb
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.lamb * self.bce_loss(output, target)


class RankLoss(nn.Module):
    
    def __init__(self, lamb=1):
        super(RankLoss, self).__init__()

        self.lamb = lamb
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.rank_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_losses, output, target):

        target_losses = self.bce_loss(output, target).detach()

        pred_losses_i, pred_losses_j = torch.split(pred_losses, 2)
        target_losses_i, target_losses_j = torch.split(target_losses, 2)

        pred_diff = pred_losses_i - pred_losses_j
        target_labels = (target_losses_i > target_losses_j).float()

        return self.lamb * self.rank_loss(pred_diff, target_labels)


class VAELoss(nn.Module):

    def __init__(self, lamb=1, beta=1):        
         super(VAELoss, self).__init__()

         self.lamb = lamb
         self.beta = beta
         self.mse_loss = nn.MSELoss()

    def forward(self, x, recon, mu, logvar):
        mse = self.mse_loss(recon, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.lamb * (mse + kld * self.beta)


class DiscLoss(nn.Module):
    
    def __init__(self, lamb=1):
        super(DiscLoss, self).__init__()

        self.lamb = lamb
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.lamb * self.bce_loss(output, target)