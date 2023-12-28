import os
import numpy as np
import argparse
import time
import random
import yaml
from dotmap import DotMap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from .dataset.datasets import get_train_loader, get_val_loader
from .models.VLCounter import Counter 
from .util import save_density_map, get_model_dir
from .test import validate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='VLCoutner')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--gpus', type=lambda s: [int(item) for item in s.split(',')], required=True, help='gpu ids')
    parser.add_argument('--enc', type=str, required=True, help='encoder setting')
    parser.add_argument('--num_tokens', type=int, help='num of SPT')
    parser.add_argument('--patch_size', type=int, required=True, help='patch size')
    parser.add_argument('--prompt', type=str, required=True, help='prompt type')
    parser.add_argument('--con', type=str, default="none", help='type of con loss')
    parser.add_argument('--exp', type=int, required=True, help='exp')
    parser.add_argument('--resume_weights', type=str, help='resume_weights')

    parsed = parser.parse_args()
    assert parsed.config is not None
    with open(parsed.config, 'r') as f:
        config = yaml.safe_load(f)
    args = DotMap(config)
    args.config = parsed.config
    args.gpus = parsed.gpus
    args.enc = parsed.enc
    args.num_tokens = parsed.num_tokens
    args.patch_size = parsed.patch_size
    args.prompt = parsed.prompt
    args.con = parsed.con
    args.exp = parsed.exp
    args.resume_weights = parsed.resume_weights

    if args.enc == 'res101':
        args.MODEL.pretrain = '/workspace/YESCUTMIX/pretrain/RN101.pt'

    model_save_dir = get_model_dir(args)
    os.makedirs(model_save_dir,exist_ok=True)
    with open(os.path.join(model_save_dir, 'FSC.yaml'), 'w') as f:
        yaml.dump(args.toDict(), f, default_flow_style=False)

    return args


def main(args):
    writer = SummaryWriter(get_model_dir(args)) # tensorboard
    
    if args.TRAIN.manual_seed is not None:
        torch.manual_seed(args.TRAIN.manual_seed)
        torch.cuda.manual_seed(args.TRAIN.manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.TRAIN.manual_seed)
        random.seed(args.TRAIN.manual_seed)
    
    #======== Metrics initialization ========
    best_mae = 1e7
    best_rmse = 1e7

    model = Counter(args).cuda()
    for k, v in model.t_enc.named_parameters():
        if 'prompt' not in k:
            v.requires_grad = False
    for k, v in model.v_enc.named_parameters():
        if 'prompt' not in k and 'text_proj' not in k and 'MHA' not in k:
            v.requires_grad=False
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    start_epoch = 0
    model_save_dir = get_model_dir(args)

    #======== Data =======
    train_loader = get_train_loader(args,mode='train')
    val_loader = get_val_loader(args,mode='val')
    
    optimizer = torch.optim.AdamW(model.parameters(),args.TRAIN.lr,weight_decay=args.TRAIN.weight_decay) # args.decay -> args.weight_decay
    # optimizer = torch.optim.AdamW([{'params': model.v_enc.parameters(), 'lr':1.0e-6}],args.TRAIN.lr, weight_decay=args.TRAIN.weight_decay)

    if args.resume_weights:
        path = os.path.join(model_save_dir, args.resume_weights)
        if os.path.isfile(path):
            print("=> loading weight '{}'".format(path))
            checkpoint = torch.load(path)
            pre_weight = checkpoint['state_dict']
            start_epoch = checkpoint['epoch'] + 1
            best_mae = checkpoint['best_mae']
            best_rmse = checkpoint['best_rmse']
            model.load_state_dict(pre_weight, strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded weight '{}'".format(path))
        else:
            raise Exception("=> no weight found at '{}'".format(args.resume_weights))


    #======== Training =======
    for epoch in range(start_epoch, args.TRAIN.epochs):
        train_loss, train_mae, train_rmse = train_model(
            args = args,
            train_loader = train_loader,
            model = model,
            optimizer = optimizer,
            epoch = epoch,
            model_save_dir = model_save_dir
        )
        writer.add_scalar('(meta-train): query loss',train_loss,epoch+1)
        writer.add_scalar('(meta-train): query mae',train_mae,epoch+1)
        writer.add_scalar('(meta-train): query rmse',train_rmse,epoch+1)
        
        val_mae, val_rmse = validate_model(
            args = args,
            val_loader = val_loader,
            model = model,
            model_save_dir = model_save_dir,
            epoch = epoch,
            mode = 'Val'
        )
        writer.add_scalar('(meta-test): query mae',val_mae,epoch+1)
        writer.add_scalar('(meta-test): query rmse',val_rmse,epoch+1)

        if best_mae >= val_mae:
            best_mae = val_mae
            best_rmse = val_rmse
            filename_model = os.path.join(model_save_dir,str(epoch)+'_'+'best.pth')
            if args.TRAIN.save_models:
                print('Saving checkpoint to: ' + filename_model)

                torch.save(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_mae' : best_mae,
                        'best_rmse' : best_rmse,
                        'optimizer': optimizer.state_dict(),
                        # 'scheduler': scheduler.state_dict()
                    },
                    filename_model
                )

        if args.TRAIN.save_models:
            filename_model = os.path.join(model_save_dir,'cur_final.pth')
            print('Saving checkpoint to: ' + filename_model)

            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_mae' : best_mae,
                    'best_rmse' : best_rmse,
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict()
                },
                filename_model
            )
        print(' * best MAE {mae:.3f} RMSE {rmse:.3f} '
                            .format(mae=best_mae,rmse=best_rmse))
        filename_model = os.path.join(model_save_dir,'final.pth')
        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_mae' : best_mae,
                'best_rmse' : best_rmse,
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict()
            },
            filename_model
        )
            

def train_model(
    args: argparse.Namespace,
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_save_dir: str
):

    model.train()
    mse_criterion = nn.MSELoss(reduction='mean').cuda()
    temp = 2
    qry_loss = 0
    qry_mae = 0
    qry_rmse = 0
    numOfQ = 0
    runtime = 0
    t0= time.time()
    for i, (query_img, query_den, tokenized_text, class_chosen) in enumerate(train_loader):
        query_img, query_den, tokenized_text = query_img.cuda().float(), query_den.cuda().float(), tokenized_text.cuda()
        numOfQ += query_img.shape[0]
        optimizer.zero_grad()

        out, attn, ori_attn = model(query_img, tokenized_text)
        # out, attn, ori_attn, tot_attn_map = model(query_img, tokenized_text)

        main = mse_criterion(out, query_den)
        
        if args.con == "none":
            infonce = 0
        elif args.con == "con":
            mask = F.interpolate(query_den, scale_factor=1/16)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = torch.where(mask>0, 1.0, 0.0)
            pos = torch.sum(torch.exp(mask * ori_attn), dim=(1,2,3))
            neg = torch.sum(torch.exp(ori_attn), dim=(1,2,3))
            
            infonce = torch.mean(-torch.log(pos / (pos + neg)))
        elif args.con == "rank":
            mask = F.interpolate(query_den, scale_factor=1/args.patch_size)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            infonce = 0
            rank = [1.0, 0.8, 0.6, 0.4]
            for i in range(3):
                r_mask = torch.where((mask > rank[i+1]) & (mask < rank[i]), 1.0, 0.0)            
                inv_mask = torch.where(mask < rank[i+1], 1.0, 0.0)
                pos = torch.sum(torch.exp(ori_attn * r_mask), dim=(1,2,3))
                neg = torch.sum(torch.exp(ori_attn * inv_mask), dim=(1,2,3))
                
                infonce += torch.mean(-torch.log(pos / (pos + neg)))
        else:
            raise NotImplementedError

        loss =  main + 0.000001 * infonce
        
        loss.backward()
        optimizer.step()


        qry_loss += loss
        out /= 60.
        query_den /= 60.
        pred_cnt = torch.sum(out, dim=(1,2,3))
        gt_cnt = torch.sum(query_den, dim=(1,2,3))
        cnt_err = abs(pred_cnt-gt_cnt)
        qry_mae += torch.sum(cnt_err).item()
        qry_rmse += torch.sum(cnt_err**2).item()
        t1= time.time()
        runtime += (t1-t0)
        if i % args.TRAIN.log_freq == 0:
            print('Epoch:[{}][{}/{}]\t'
                  'main/MAE/RMSE [{},{:5.3f},{:5.3f}]\t'
                  'Runtime:[{}]'
                  .format(epoch, i,len(train_loader),
                          main, qry_mae/numOfQ, (qry_rmse/numOfQ)**0.5,
                          runtime/numOfQ),flush=True)
            visualize_path = model_save_dir + "/visualize_train"
            os.makedirs(visualize_path,exist_ok=True)
            # save_density_map(query_img[0],out[0],attn[0],query_den[0,0], visualize_path, str(epoch)+'_'+str(i),class_chosen=class_chosen[0])
            # save_density_map(query_img[0],out1[0],attn1[0],query_den[0,1], visualize_path, str(epoch)+'_'+str(i)+'cutmix',class_chosen=shuffled_class_name[0])

    qry_loss = qry_loss / len(train_loader)
    qry_mae = qry_mae / len(train_loader.dataset) 
    qry_rmse = (qry_rmse / len(train_loader.dataset)) ** 0.5
    print('Epoch {}: Loss/MAE/RMSE/AVG runtime [{:5.5f},{:5.3f},{:5.3f},{}]'.format(
        epoch, qry_loss,qry_mae,qry_rmse,runtime/len(train_loader)))
    print("Epoch runtime {}".format(time.time() - runtime))
    return qry_loss, qry_mae, qry_rmse

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    main(args)