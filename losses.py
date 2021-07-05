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

        pred_losses_i, pred_losses_j = torch.chunk(pred_losses, 2)
        target_losses_i, target_losses_j = torch.chunk(target_losses, 2)

        pred_diff = pred_losses_i - pred_losses_j
        target_labels = (target_losses_i > target_losses_j).float()

        return self.lamb * self.rank_loss(pred_diff, target_labels)


class VAELoss(nn.Module):

    def __init__(self, lamb=1, trd=1, adv=10, beta=1):        
         super(VAELoss, self).__init__()

         self.lamb = lamb
         self.trd = trd
         self.adv = adv
         self.beta = beta
         self.mse_loss = nn.MSELoss()
         self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, recon, mu, logvar, disc_output):
        mse = self.mse_loss(recon, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        trd_loss = mse + self.beta * kld
        adv_loss = self.bce_loss(disc_output, torch.ones_like(disc_output))
        vae_loss = self.trd * trd_loss + self.adv * adv_loss
        return self.lamb * vae_loss


class DiscLoss(nn.Module):
    
    def __init__(self, lamb=1):
        super(DiscLoss, self).__init__()

        self.lamb = lamb
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.lamb * self.bce_loss(output, target)