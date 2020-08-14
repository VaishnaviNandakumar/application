import numpy as np
import torch
from torch import nn
import argparse
import time
import os
import sys
import utils.Earlystopping
from utils.onecycle import OneCycle
from utils.util import *
import argparse
from core.losses import loss
from torchvision import models
from core.optimizer import optimizers
from config import get_cfg_defaults
import wandb
from dataset.datasetcClass import classDataset
from torch.utils.data import DataLoader
from model import Classifier

# /**
# * TODO: 1. Add code for model debugging and visualization
# * TODO: 2. Add code for confusion matrix plot
# * TODO: 3. Add learning rate schedulers
# * TODO: 4. Add resume capability
# */

def train_epoch(epoch, data_loader, model, criterion , optimizer , cfg):
    
    """
    Batch training loop
    Arguments:
        epoch {[int]} -- [current epoch number]
        data_loader {[type]} -- [pytorch train dataloader]
        model {[type]} -- [pytorch model]
        criterion {[type]} -- [loss function]
        optimizer {[type]} -- [optimizer]
        cfg {[type]} -- [config parameters required]
    """
        
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    for i, sample in enumerate(data_loader):
        inputs = sample['image'].type(torch.FloatTensor)
        if(i == 0):
            wandb.log({"examples": [wandb.Image(wandb_plot(inputs[0],cfg), caption="Label")]})
        targets = sample['label']
        if torch.cuda.is_available():
            targets = targets.cuda()
            inputs = inputs.cuda()           
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        
        #add different learning rate schedulers
        # if(cfg.train.onecycle):
        #     lr,_ = onecyc.calc()
        #     update_lr(optimizer, lr)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d / %d] [Loss: %f, Acc: %.2f%%]"
                % (
                    epoch,
                    cfg.train.n_epochs,
                    i,
                    len(data_loader),
                    losses.avg,
                    accuracies.avg
                    )
                )
        wandb.log({'Batch accuracy': accuracies.val,'batch Loss':losses.val,'lr': optimizer.param_groups[0]['lr']})
        
    save_file_path = os.path.join(cfg.train.ckpt_save_dir,'{}_{}_{}.pth'.format(cfg.model.backbone,cfg.train.config_path.split('/')[-1][:-5],epoch)) #Add os.path.join based on epoch number and upload files on wandb
    
    states = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    print('Epoch accuracy' , accuracies.avg,'Epoch Loss' , losses.avg)
    wandb.log({'Epoch accuracy': accuracies.avg,'Epoch Loss':losses.avg})
    torch.save(states, save_file_path)
    wandb.save(save_file_path)
    
def validate_model(epoch,model, data_loader ,criterion):
    """
    To validate model on valid dataset
    Arguments:
        epoch {[int]} -- [epoch number will use for global step in wandb]
        model {[pytorch model]} -- [model to validate]
        data_loader {[pytorch dataloader]} -- [valid loader]
        criterion {[pytorch loss function]} -- [loss function]
    """    
    
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            inputs = sample['image'].type(torch.FloatTensor)
            targets = sample['label']
            if torch.cuda.is_available():
                targets = targets.cuda()
                inputs = inputs.cuda()
            
            outputs = model(inputs)
            loss = torch.mean(los(outputs, targets.type(torch.cuda.LongTensor)))
            acc = calculate_accuracy(outputs,targets.type(torch.cuda.LongTensor))
            
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            
            sys.stdout.write(
                    "\r[Batch %d / %d]  [Loss: %f, Acc: %.2f%%]"
                    % (
                        i,
                        len(data_loader),
                        losses.avg,
                        accuracies.avg
                        )
                    )
    wandb.log({'Validation accuracy': accuracies.avg,'Validation Loss':losses.avg})
    return true,pred,best_acc

def main():
    ####################argument parser#################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="",
                        required= True,
                        help='Location of current config file')
    parser.add_argument('--dataset_csvpath', type=str, required=True, default="./",
                        help='Location to data csv file')
    
    parser.add_argument("--wandbkey", type=str,
                        default='2d5e5aa07e2a9cd4f84004f838566b5eca9f5856',
                        help='Wandb project key')
    parser.add_argument("--wandbproject", type=str, required = True,
                        default='',
                        help='wandb project name')
    parser.add_argument("--wandbexperiment", type=str,required = True,
                        default='',
                        help='wandb experiment name')
    
    parser.add_argument("--ckpt_save_dir", type=str, default='./ckpts',
                        help='path to save checkpoints')
        
    parser.add_argument("--resume_checkpoint_path", type = str, default = '',
                        help='If you want to resume training enter path to checkpoint')
    
    #####################read config file###############################
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    cfg.merge_from_list(['train.config_path',args.config_path,'dataset.csvpath',args.dataset_csvpath,'train.ckpt_save_dir',args.ckpt_save_dir])
    cfg.freeze()
    print(cfg)
    ####### Wandb
    os.makedirs(args.ckpt_save_dir,exist_ok = True)
    os.system('wandb login {}'.format(args.wandbkey))
    wandb.init(name = args.wandbexperiment,project= args.wandbproject,config = cfg)
    wandb.save(args.config_path) # Save configuration file on wandb

    ################################################################
    validate = False
    if(os.path.exists(os.path.join(args.dataset_csvpath,'valid.csv'))):
        validate = True
    train_object = classDataset(cfg,'train')
    train_loader = DataLoader(train_object,
                              batch_size = cfg.dataset.batch_size_pergpu*len(cfg.train.gpus), 
                              shuffle = cfg.dataset.shuffle, 
                              num_workers = cfg.dataset.num_workers)
    
    if(validate):
        valid_object = classDataset(cfg,'valid')
        valid_loader = DataLoader(valid_object,
                                batch_size = cfg.dataset.batch_size_pergpu*len(cfg.train.gpus), 
                                shuffle = cfg.dataset.shuffle, 
                                num_workers = cfg.dataset.num_workers)
    
    ################################################################

    model = Classifier(cfg)
    if(torch.cuda.is_available()):
        model.cuda()
         
    criterion = loss[cfg.Loss.val](cfg)
    optimizer = optimizers[cfg.optimizer.val](model.parameters(),cfg)
    start_epoch = 1
    
    if(args.resume_checkpoint_path != ''):
        old_dict = torch.load(args.resume_checkpoint_path)
        model.load_state_dict(old_dict['state_dict'])
        optimizer.load_state_dict(old_dict['optim_dict'])
        start_epoch = old_dict['epoch']
        
    #################################################################
    
    for epoch in range(start_epoch,cfg.train.n_epochs):
        train_epoch(epoch, train_loader, model,criterion,optimizer,cfg)
        if(validate):
            if(epoch%cfg.valid.frequency):
                validate_model(epoch,model.valid_loader,criterion)
                
    #Fine tune model with very low learning rate Add example
    if(cfg.train.tune.val):
        set_lr(optimizer,lr = cfg.optimizer.lr/cfg.train.tune.lr_factor)
        train_epoch(epoch + 1, train_loader, model , criterion , optimizer, cfg)
if __name__ == '__main__':
    main()
