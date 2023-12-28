import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import random
import yaml
from dotmap import DotMap
import scipy.ndimage as ndimage

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from .models.Counter_vit_af_tc_info_unet_v4 import Counter 
from .models.Counter_vit_tc_unet_info import Counter 
from .util import save_density_map, save_density_map_carpk, get_model_dir, get_model_dir_carpk
def parse_args() -> None:
    parser = argparse.ArgumentParser(description='Zero Shot Object Counting')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--gpus', type=lambda s: [int(item) for item in s.split(',')], required=True, help='gpu ids')
    parser.add_argument('--enc', type=str, required=True, help='LIT encoder setting')
    parser.add_argument('--prompt', type=str, required=True, help='num of prompt')
    parser.add_argument('--ckpt_used', type=str, required=True, help='best checkpoint')
    parser.add_argument('--exp', type=int, required=True, help='exp')

    parsed = parser.parse_args()
    assert parsed.config is not None
    with open(parsed.config, 'r') as f:
        config = yaml.safe_load(f)
    args = DotMap(config)
    args.config = parsed.config
    args.gpus = parsed.gpus
    args.enc = parsed.enc
    args.prompt = parsed.prompt
    args.EVALUATION.ckpt_used = parsed.ckpt_used
    args.exp = parsed.exp

    if args.enc == 'res101':
        args.MODEL.pretrain = '/workspace/YESCUTMIX/pretrain/RN101.pt'
    
    return args


def main(args):
    local_rank = args.local_rank

    if args.TRAIN.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.TRAIN.manual_seed)
        np.random.seed(args.TRAIN.manual_seed)
        torch.manual_seed(args.TRAIN.manual_seed)
        torch.cuda.manual_seed_all(args.TRAIN.manual_seed)
        random.seed(args.TRAIN.manual_seed)
    
    model = Counter(args).cuda()
    root_model = get_model_dir(args)

    if args.EVALUATION.ckpt_used is not None:
        filepath = os.path.join(root_model, f'{args.EVALUATION.ckpt_used}.pth')
        assert os.path.isfile(filepath), filepath
        print("=> loading model weight '{}'".format(filepath),flush=True)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model weight '{}'".format(filepath),flush=True)
    else:
        print("=> Not loading anything",flush=True)
    
    # test_loader = get_val_loader(args,mode='test')
    
    import hub
    ds_test = hub.load("hub://activeloop/carpk-test")
    #dataloader_train = ds_train.pytorch(num_workers=args.num_workers, batch_size=1, shuffle=False)
    test_loader = ds_test.pytorch(num_workers=args.DATA.workers, batch_size=1, shuffle=False)
    
    root_model = get_model_dir_carpk(args)

    # ====== Test  ======
    val_mae,val_rmse =validate_model(
        args= args,
        val_loader=test_loader,
        model=model,
        model_save_dir=os.path.join(root_model,'inference'),
        epoch = 0,
        mode = 'test'
    )


def validate_model(
    args: argparse.Namespace,
    val_loader: torch.utils.data.DataLoader,
    model : torch.nn.Module,
    model_save_dir : str,
    epoch : int,
    mode : str
):
    multi_plural_prompt_templates = ['a photo of a number of {}.',
                                'a photo of a number of small {}.',
                                'a photo of a number of medium {}.',
                                'a photo of a number of large {}.',
                                'there are a photo of a number of {}.',
                                'there are a photo of a number of small {}.',
                                'there are a photo of a number of medium {}.',
                                'there are a photo of a number of large {}.',
                                'a number of {} in the scene.',
                                'a photo of a number of {} in the scene.',
                                'there are a number of {} in the scene.',
                                ]

    print("===> Start testing")

    model.eval()
    mse_criterion = nn.MSELoss(reduction='mean').cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()
    qry_loss = 0
    qry_mae = 0
    qry_rmse = 0
    runtime = 0
    already_print_class = []
    from PIL import Image
    from torchvision import transforms
    img_trans = transforms.Compose([
                        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
                ])
    sort_err = dict()
    with torch.no_grad():
        flag=0
        text = [template.format("cars") for template in multi_plural_prompt_templates]

        from .tokenizer import tokenize
        tokenized_text = tokenize(text)
        tokenized_text = tokenized_text.cuda().unsqueeze(0)
        for idx, sample in enumerate(val_loader):
            query_img, labels = (sample['images']/255).cuda().float(), sample['labels'].cuda()
            query_img = query_img.transpose(2,3).transpose(1,2)
            query_img = transforms.Resize((384, 683))(query_img[0])
            query_img = img_trans(query_img).unsqueeze(0)
            t0= time.time()

            density_map = torch.zeros([384,683])
            density_map = density_map.cuda()
            attn_map = torch.zeros([384,683])
            attn_map = attn_map.cuda()
            pred_cnt = 0
            start = 0
            prev = -1
            while start + 383 < 683:
                output, attn, _ = model(query_img[:,:,:,start:start+384], tokenized_text, None)
                output = output.squeeze(0).squeeze(0)
                attn = attn.squeeze()
                b1 = nn.ZeroPad2d(padding=(start, 683 - prev - 1, 0, 0))
                d1 = b1(output[:, 0:prev - start + 1])
                a1 = b1(attn[:, 0:prev - start + 1])
                b2 = nn.ZeroPad2d(padding=(prev + 1, 683 - start - 384, 0, 0))
                d2 = b2(output[:, prev - start + 1:384])
                a2 = b2(attn[:, prev - start + 1:384])

                b3 = nn.ZeroPad2d(padding=(0, 683 - start, 0, 0))
                density_map_l = b3(density_map[:, 0:start])
                density_map_m = b1(density_map[:, start:prev + 1])
                attn_map_l = b3(attn_map[:, 0:start])
                attn_map_m = b1(attn_map[:, start:prev + 1])
                b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                density_map_r = b4(density_map[:, prev + 1:683])
                attn_map_r = b4(attn_map[:, prev + 1:683])
                
                
                density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                attn_map = attn_map_l + attn_map_r + attn_map_m / 2 + a1 / 2 + a2

                prev = start + 383
                start = start + 128
                if start+383 >= 683:
                    if start == 683 - 384 + 128: break
                    else: start = 683 - 384
        
            conv = nn.Conv2d(1,1,kernel_size=(16,16),stride=16,bias=False)
            #print(conv.weight.shape)
            conv.weight.data = torch.ones([1,1,16,16]).cuda()

            density_map = density_map.unsqueeze(0)
            density_map = density_map.unsqueeze(0)
            d_m = conv(density_map/60)
            pred_cnt += torch.sum(d_m).item()
            for i in range(d_m.shape[2]):
                for j in range(d_m.shape[3]):
                    if d_m[0][0][i][j] > 1.224:
                        pred_cnt -=1
            
            gt_cnt = labels.shape[1]
            cnt_err = abs(pred_cnt-gt_cnt)
            qry_mae += cnt_err
            qry_rmse += cnt_err**2
            sort_err[idx] = cnt_err
            # gt_density = np.zeros((384, 683),dtype='float32')
            # for box in sample['boxes'][0]:
            #     box2 = [int(k) for k in box]
            #     x, y = int(box2[0]+box2[2]/2), int(box2[1]+box2[3]/2)
            #     # if x < 720:
            #     x = int(x * 683 / 1280)
            #     y = int(y * 384 / 720)
            #     gt_density[y][x] = 1
            # gt_density = ndimage.gaussian_filter(gt_density, sigma=(1, 1), order=0)
            # gt_density = torch.from_numpy(gt_density)
            # gt_density = gt_density.cuda()

            # visualize_path = model_save_dir + "/visualize_test_carpk"
            # os.makedirs(visualize_path,exist_ok=True)
            # save_density_map(query_img[0],density_map,attn_map,gt_density,visualize_path,str(idx)+'_cars',class_chosen="cars")
            # save_density_map_carpk(query_img[0],density_map,attn_map,cnt_err,visualize_path,str(idx)+'_cars',class_chosen="cars")
        qry_mae = qry_mae / len(val_loader.dataset)
        qry_rmse = (qry_rmse/len(val_loader.dataset)) ** 0.5
    for w in sorted(sort_err, key=sort_err.get, reverse=True):
        print(w, sort_err[w])
    print("Test result: MAE/RMSE [{:5.2f},{:5.2f}]".format(qry_mae, qry_rmse))

    return qry_mae, qry_rmse

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    main(args)