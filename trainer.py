import os
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from scipy.special import expit
from itertools import cycle


def batch_acc(y_preds, y_targets):
    y_true = y_targets.cpu().detach().numpy()
    y_pred = (expit(y_preds.cpu().detach().numpy()) > 0.5).astype(int)

    return (y_true == y_pred).mean(axis=0)


def run_train(
    save_dir,
    task_model,
    vae,
    disc,
    labeled_train_loader,
    labeled_val_loader,
    unlabeled_train_loader,
    unlabeled_val_loader,
    task_loss,
    rank_loss,
    vae_loss,
    disc_loss,
    task_model_optimizer,
    vae_optimizer,
    disc_optimizer,
    epochs,
    task_lr_scheduler=None,
    task_lr_scheduler_target='val_loss',
    vae_lr_scheduler=None,
    vae_lr_scheduler_target='val_loss',
    disc_lr_scheduler=None,
    disc_lr_scheduler_target='val_loss',
    train_patient=None,
    val_patient=None,
    device='cuda',
    grad_scaler=None,
):

    unlabeled_train_iter = cycle(unlabeled_train_loader)
    unlabeled_val_iter = cycle(unlabeled_val_loader)

    train_loss_rec = []
    train_task_loss_rec = []
    train_vae_loss_rec = []
    train_disc_loss_rec = []
    train_acc_rec = []
    cur_train_patient = train_patient
    train_best_loss = np.inf

    val_loss_rec = []
    val_task_loss_rec = []
    val_vae_loss_rec = []
    val_disc_loss_rec = []
    val_acc_rec = []
    cur_val_patient = val_patient
    val_best_loss = np.inf

    for epoch in range(1, epochs + 1):

        if cur_train_patient == 0 or cur_val_patient == 0:
            print('===== Early stop was reached =====')
            break

        new_best = False

        batch_train_loss_rec = []
        batch_train_task_loss_rec = []
        batch_train_vae_loss_rec = []
        batch_train_disc_loss_rec = []
        batch_train_acc_rec = []

        for batch_labeled_train_X, batch_labeled_train_y in tqdm(labeled_train_loader):
            batch_unlabeled_train_X, _ = next(unlabeled_train_iter)
            batch_train_task_loss, batch_train_vae_loss, batch_train_disc_loss, batch_train_acc = train(
                task_model,
                vae,
                disc,
                batch_labeled_train_X,
                batch_labeled_train_y,
                batch_unlabeled_train_X,
                task_loss,
                rank_loss,
                vae_loss,
                disc_loss,
                task_model_optimizer,
                vae_optimizer,
                disc_optimizer,
                device,
                grad_scaler,
                batch_acc
            )
            batch_train_loss = sum([batch_train_task_loss, batch_train_vae_loss, batch_train_disc_loss])
            batch_train_loss_rec.append(batch_train_loss)
            batch_train_task_loss_rec.append(batch_train_task_loss)
            batch_train_vae_loss_rec.append(batch_train_vae_loss)
            batch_train_disc_loss_rec.append(batch_train_disc_loss)
            batch_train_acc_rec.append(batch_train_acc)

        train_loss_rec.append(np.mean(batch_train_loss_rec))
        train_task_loss_rec.append(np.mean(batch_train_task_loss_rec))
        train_vae_loss_rec.append(np.mean(batch_train_vae_loss_rec))
        train_disc_loss_rec.append(np.mean(batch_train_disc_loss_rec))
        train_acc_rec.append(np.mean(np.stack(batch_train_acc_rec), axis=0))

        batch_val_loss_rec = []
        batch_val_task_loss_rec = []
        batch_val_vae_loss_rec = []
        batch_val_disc_loss_rec = []
        batch_val_acc_rec = []

        for batch_labeled_val_X, batch_labeled_val_y in labeled_val_loader:
            batch_unlabeled_val_X, _ = next(unlabeled_val_iter)
            batch_val_task_loss, batch_val_vae_loss, batch_val_disc_loss, batch_val_acc = evaluate(
                task_model,
                vae,
                disc,
                batch_labeled_val_X,
                batch_labeled_val_y,
                batch_unlabeled_val_X,
                task_loss,
                rank_loss,
                vae_loss,
                disc_loss,
                device,
                batch_acc)

            batch_val_loss = sum([batch_val_task_loss, batch_val_vae_loss, batch_val_disc_loss])
            batch_val_loss_rec.append(batch_val_loss)
            batch_val_task_loss_rec.append(batch_val_task_loss)
            batch_val_vae_loss_rec.append(batch_val_vae_loss)
            batch_val_disc_loss_rec.append(batch_val_disc_loss)
            batch_val_acc_rec.append(batch_val_acc)

        val_loss_rec.append(np.mean(batch_val_loss_rec))
        val_task_loss_rec.append(np.mean(batch_val_task_loss_rec))
        val_vae_loss_rec.append(np.mean(batch_val_vae_loss_rec))
        val_disc_loss_rec.append(np.mean(batch_val_disc_loss_rec))
        val_acc_rec.append(np.mean(np.stack(batch_val_acc_rec), axis=0))

        if (task_lr_scheduler is not None) and (task_lr_scheduler_target is not None):

            if task_lr_scheduler_target == 'val_loss':
                task_lr_schd_target = val_task_loss_rec[-1]

            elif task_lr_scheduler_target in ('loss', 'train_loss'):
                task_lr_schd_target = train_task_loss_rec[-1]

            task_lr_scheduler.step(task_lr_schd_target)

        if (vae_lr_scheduler is not None) and (vae_lr_scheduler_target is not None):

            if vae_lr_scheduler_target == 'val_loss':
                vae_lr_schd_target = val_vae_loss_rec[-1]

            elif vae_lr_scheduler_target in ('loss', 'train_loss'):
                vae_lr_schd_target = train_vae_loss_rec[-1]

            vae_lr_scheduler.step(vae_lr_schd_target)

        if (disc_lr_scheduler is not None) and (disc_lr_scheduler_target is not None):

            if disc_lr_scheduler_target == 'val_loss':
                disc_lr_schd_target = val_disc_loss_rec[-1]

            elif disc_lr_scheduler_target in ('loss', 'train_loss'):
                disc_lr_schd_target = train_disc_loss_rec[-1]

            disc_lr_scheduler.step(disc_lr_schd_target)

        if train_loss_rec[-1] > train_best_loss:
            if train_patient is not None:
                cur_train_patient -= 1

        else:
            train_best_loss = train_loss_rec[-1]
            cur_train_patient = train_patient

        if val_loss_rec[-1] > val_best_loss:
            if val_patient is not None:
                cur_val_patient -= 1

        elif val_loss_rec[-1] >= train_loss_rec[-1]:
            val_best_loss = val_loss_rec[-1]
            cur_val_patient = val_patient

            # to extend training
            train_best_loss = train_loss_rec[-1]
            cur_train_patient = train_patient

            prev_checkpoints = os.listdir(save_dir)
            for prev_checkpoint in prev_checkpoints:
                prev_checkpoint_path = os.path.join(save_dir, prev_checkpoint)
                if prev_checkpoint_path.endswith('.tar'):
                    os.remove(prev_checkpoint_path)

            new_best = True

        print('epoch {}: train_loss = {:.4f},    valid_loss = {:.4f}\
            \n         train_task_loss = {:.4f},    valid_task_loss = {:.4f}\
            \n         train_vae_loss = {:.4f},    valid_vae_loss = {:.4f}\
            \n         train_disc_loss = {:.4f},    valid_disc_loss = {:.4f}\
            \n         train_slice_acc = {:.4f},    valid_slice_acc = {:.4f}'.format(
            epoch,
            train_loss_rec[-1],
            val_loss_rec[-1],
            train_task_loss_rec[-1],
            val_task_loss_rec[-1],
            train_vae_loss_rec[-1],
            val_vae_loss_rec[-1],
            train_disc_loss_rec[-1],
            val_disc_loss_rec[-1],
            train_acc_rec[-1][0],
            val_acc_rec[-1][0],
        ))

        if new_best or epoch == epochs or cur_train_patient == 0 or cur_val_patient == 0:
            torch.save({
                'epoch': epoch,
                'cur_train_patient': cur_train_patient,
                'cur_val_patient': cur_val_patient,
                'task_model_state_dict': task_model.state_dict(),
                'vae_state_dict': vae.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'task_model_optimizer_state_dict': task_model_optimizer.state_dict(),
                'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                'task_lr_scheduler_state_dict': task_lr_scheduler.state_dict() if task_lr_scheduler is not None else None,
                'vae_lr_scheduler_state_dict': vae_lr_scheduler.state_dict() if vae_lr_scheduler is not None else None,
                'disc_lr_scheduler_state_dict': disc_lr_scheduler.state_dict() if disc_lr_scheduler is not None else None,
                'grad_scaler_state_dict': grad_scaler.state_dict() if grad_scaler is not None else None,
                'train_loss_rec': train_loss_rec,
                'val_loss_rec': val_loss_rec,
                'train_task_loss_rec': train_task_loss_rec,
                'val_task_loss_rec': val_task_loss_rec,
                'train_vae_loss_rec': train_vae_loss_rec,
                'val_vae_loss_rec': val_vae_loss_rec,
                'train_disc_loss_rec': train_disc_loss_rec,
                'val_disc_loss_rec': val_disc_loss_rec,
                'train_acc_rec': train_acc_rec,
                'val_acc_rec': val_acc_rec,
            }, os.path.join(save_dir, 'check_point_{}_loss_{:.4f}.tar'.format(epoch, val_loss_rec[-1])))


def train(
    task_model,
    vae,
    disc,
    labeled_X,
    labeled_y,
    unlabeled_X,
    task_criterion,
    rank_criterion,
    vae_criterion,
    disc_criterion,
    task_model_optimizer,
    vae_optimizer,
    disc_optimizer,
    device='cuda',
    grad_scaler=None,
    metric=None,
    ):

    task_model.train()
    vae.train()
    disc.train()

    labeled_X, labeled_y, unlabeled_X = labeled_X.to(device), labeled_y.to(device), unlabeled_X.to(device)
    labeled_factor = len(labeled_X) / (len(labeled_X) + len(unlabeled_X))
    unlabeled_factor = 1 - labeled_factor

    task_model_optimizer.zero_grad()
    vae_optimizer.zero_grad()

    # labeled data
    with autocast(enabled=grad_scaler is not None):
        task_output_l = task_model(labeled_X)
        loss_rank_l = torch.sigmoid(task_output_l['loss_pred'])
        recon_l, z_r_l, mu_l, logvar_l = vae(labeled_X, loss_rank_l)
        disc_output_l = disc(z_r_l)
        
        task_loss = task_criterion(task_output_l['pred'], labeled_y)
        rank_loss = rank_criterion(task_output_l['loss_pred'], task_output_l['pred'], labeled_y)
        labeled_vae_loss = vae_criterion(labeled_X, recon_l, mu_l, logvar_l, disc_output_l) * labeled_factor

    if grad_scaler is not None:
        grad_scaler.scale(task_loss).backward(retain_graph=True)
        grad_scaler.scale(rank_loss).backward(retain_graph=True)
        grad_scaler.scale(labeled_vae_loss).backward(retain_graph=True)
    
    else:
        task_loss.backward(retain_graph=True)
        rank_loss.backward(retain_graph=True)
        labeled_vae_loss.backward(retain_graph=True)
       
    # unlabeled data
    with autocast(enabled=grad_scaler is not None):
        task_output_u = task_model(unlabeled_X)
        loss_rank_u = torch.sigmoid(task_output_u['loss_pred'])
        recon_u, z_r_u, mu_u, logvar_u = vae(unlabeled_X, loss_rank_u)
        disc_output_u = disc(z_r_u)
        
        unlabeled_vae_loss = vae_criterion(unlabeled_X, recon_u, mu_u, logvar_u, disc_output_u) * unlabeled_factor
        
    if grad_scaler is not None:
        grad_scaler.scale(unlabeled_vae_loss).backward(retain_graph=True)

    else:
        unlabeled_vae_loss.backward(retain_graph=True)

    # discriminator
    disc_optimizer.zero_grad()

    # labeled data
    with autocast(enabled=grad_scaler is not None):

        with torch.no_grad():
            _, z_r_l, _, _ = vae(labeled_X, loss_rank_l)

        disc_output_l = disc(z_r_l)    
        labeled_disc_loss = disc_criterion(disc_output_l, torch.ones_like(disc_output_l)) * labeled_factor
    
    if grad_scaler is not None:
        grad_scaler.scale(labeled_disc_loss).backward()
    
    else:
        labeled_disc_loss.backward()

    # unlabeled data
    with autocast(enabled=grad_scaler is not None):

        with torch.no_grad():
            _, z_r_u, _, _ = vae(unlabeled_X, loss_rank_u)

        disc_output_u = disc(z_r_u)
        unlabeled_disc_loss = disc_criterion(disc_output_u, torch.zeros_like(disc_output_u)) * unlabeled_factor

    if grad_scaler is not None:
        grad_scaler.scale(unlabeled_disc_loss).backward()

        grad_scaler.step(task_model_optimizer)
        grad_scaler.step(vae_optimizer)
        grad_scaler.step(disc_optimizer)

        grad_scaler.update()

    else:
        unlabeled_disc_loss.backward()

        task_model_optimizer.step()
        vae_optimizer.step()
        disc_optimizer.step()

    total_task_loss = task_loss + rank_loss
    vae_loss = labeled_vae_loss + unlabeled_vae_loss
    disc_loss = labeled_disc_loss + unlabeled_disc_loss

    if metric is not None:
        return total_task_loss.item(), vae_loss.item(), disc_loss.item(), metric(task_output_l['pred'], labeled_y)

    return total_task_loss.item(), vae_loss.item(), disc_loss.item()


@torch.no_grad()
def evaluate(
    task_model,
    vae,
    disc,
    labeled_X,
    labeled_y,
    unlabeled_X,
    task_criterion,
    rank_criterion,
    vae_criterion,
    disc_criterion,
    device='cuda',
    metric=None,
    ):

    task_model.eval()
    vae.eval()
    disc.eval()

    labeled_X, labeled_y, unlabeled_X = labeled_X.to(device), labeled_y.to(device), unlabeled_X.to(device)
    labeled_factor = len(labeled_X) / (len(labeled_X) + len(unlabeled_X))
    unlabeled_factor = 1 - labeled_factor

    # labeled data
    task_output_l = task_model(labeled_X)
    loss_rank_l = torch.sigmoid(task_output_l['loss_pred'])
    recon_l, z_r_l, mu_l, logvar_l = vae(labeled_X, loss_rank_l)
    disc_output_l = disc(z_r_l)
    
    task_loss = task_criterion(task_output_l['pred'], labeled_y)
    rank_loss = rank_criterion(task_output_l['loss_pred'], task_output_l['pred'], labeled_y)
    labeled_vae_loss = vae_criterion(labeled_X, recon_l, mu_l, logvar_l, z_r_l) * labeled_factor
    labeled_disc_loss = disc_criterion(disc_output_l, torch.ones_like(disc_output_l)) * labeled_factor

    # unlabeled data
    task_output_u = task_model(unlabeled_X)
    loss_rank_u = torch.sigmoid(task_output_u['loss_pred'])
    recon_u, z_r_u, mu_u, logvar_u = vae(unlabeled_X, loss_rank_u)
    disc_output_u = disc(z_r_u)
    
    unlabeled_vae_loss = vae_criterion(unlabeled_X, recon_u, mu_u, logvar_u, z_r_u) * unlabeled_factor
    unlabeled_disc_loss = disc_criterion(disc_output_u, torch.zeros_like(disc_output_u)) * unlabeled_factor
    
    total_task_loss = task_loss + rank_loss
    vae_loss = labeled_vae_loss + unlabeled_vae_loss
    disc_loss = labeled_disc_loss + unlabeled_disc_loss

    if metric is not None:
        return total_task_loss.item(), vae_loss.item(), disc_loss.item(), metric(task_output_l['pred'], labeled_y)

    return total_task_loss.item(), vae_loss.item(), disc_loss.item()

