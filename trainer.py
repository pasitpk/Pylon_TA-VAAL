import os
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from scipy.special import expit


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
    lr_scheduler=None,
    lr_scheduler_target='val_loss',
    train_patient=None,
    val_patient=None,
    device='cuda',
    grad_scaler=None,
    ):
    
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

        for batch_train_X, batch_train_y in tqdm(train_loader):
            batch_train_loss, batch_train_acc = train(model, batch_train_X, batch_train_y, criterion_train, optimizer, device, grad_scaler, batch_acc)
            batch_train_loss_rec.append(batch_train_loss)
            batch_train_acc_rec.append(batch_train_acc)

        train_loss_rec.append(np.mean(batch_train_loss_rec))
        train_acc_rec.append(np.mean(np.stack(batch_train_acc_rec), axis=0))

        batch_val_loss_rec = []
        batch_val_task_loss_rec = []
        batch_val_vae_loss_rec = []
        batch_val_disc_loss_rec = []
        batch_val_acc_rec = []

        for batch_val_X, batch_val_y in val_loader:

            batch_val_loss, batch_val_acc = evaluate(model, batch_val_X, batch_val_y, criterion_val, device, batch_acc)

            batch_val_loss_rec.append(batch_val_loss)
            batch_val_acc_rec.append(batch_val_acc)

        val_loss_rec.append(np.mean(batch_val_loss_rec))
        val_acc_rec.append(np.mean(np.stack(batch_val_acc_rec), axis=0))

        if (lr_scheduler is not None) and (lr_scheduler_target is not None):

            if lr_scheduler_target == 'val_loss':
                lr_schd_target = val_loss_rec[-1]

            elif lr_scheduler_target in ('loss', 'train_loss'):
                lr_schd_target = train_loss_rec[-1]

            lr_scheduler.step(lr_schd_target)
        
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
                        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
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


def train(model, X, y, criterion, optimizer, device='cuda', grad_scaler=None, metric=None):

    model.train()
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()

    if grad_scaler is not None:

        with autocast():
            output = model(X)['pred']
            loss = criterion(output, y)

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

    else:
        output = model(X)['pred']
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()
        
    if metric is not None:
        return loss.item(), metric(output, y)

    return loss.item()


@torch.no_grad()
def evaluate(model, X, y, criterion, device='cuda', metric=None):

    model.eval()

    X, y = X.to(device), y.to(device)
    output = model(X)['pred']

    loss = criterion(output, y).item()

    model.train()
    
    if metric is not None:
        return loss, metric(output, y)

    return loss