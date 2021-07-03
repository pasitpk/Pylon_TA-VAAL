import torch
from torch import nn


class TaskLoss(nn.Module):
    
    def __init__(self):
        super(TaskLoss, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.bce_loss(output, target)


class RankLoss(nn.Module):
    
    def __init__(self):
        super(RankLoss, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_losses, target_losses):

        pred_losses_i, pred_losses_j = torch.split(pred_losses, 2)
        target_losses_i, target_losses_j = torch.split(target_losses, 2)

        pred_diff = pred_losses_i - pred_losses_j
        target_labels = (target_losses_i > target_losses_j).float()

        return self.bce_loss(pred_diff, target_labels)


class VAELoss(nn.Module):

    def __init__(self, beta=1):        
         super(VAELoss, self).__init__()

         self.beta = beta
         self.mse_loss = nn.MSELoss()

    def forward(self, x, recon, mu, logvar):
        mse = self.mse_loss(recon, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld * self.beta


class DiscLoss(nn.Module):
    
    def __init__(self):
        super(DiscLoss, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.bce_loss(output, target)