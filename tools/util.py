import os
import argparse
import numpy as np
import cv2


def save_density_map_carpk(query_img, pred_D,attn,cnt_err,output_dir, fname='results.png', class_chosen=None, pred_cnt=None):

    if query_img is not None:
        _,h,w = query_img.shape
        query_img = query_img.cpu().numpy()
        query_img = 255.0 * (query_img - np.min(query_img) + 1e-10) / (1e-10 + np.max(query_img) - np.min(query_img))
        query_img = query_img.squeeze()
        query_img = query_img.transpose(1,2,0)
        query_img = cv2.cvtColor(query_img,cv2.COLOR_BGR2RGB)
    
    if pred_D is not None:
        pred_D = pred_D.cpu().detach().numpy()
        if pred_cnt is None:
            pred_cnt = np.sum(pred_D)
        pred_D = 255.0 * (pred_D - np.min(pred_D) + 1e-10) / (1e-10 + np.max(pred_D) - np.min(pred_D))
        pred_D = pred_D.squeeze()
        pred_D = cv2.applyColorMap(pred_D[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)
    
    if attn is not None:
        attn = attn.cpu().detach().numpy()
        attn = 255.0 * (attn - np.min(attn) + 1e-10) / (1e-10 + np.max(attn) - np.min(attn))
        attn = attn.squeeze()
        attn = cv2.applyColorMap(attn[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)

    # h,w = pred_D.shape[:2]
    # cv2.putText(query_img,class_chosen,(0,20),cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,0), 1)
    # if query_img is not None:
    #     cv2.putText(query_img,"Den Predict", (0,20), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)
    #     cv2.putText(query_img,str(cnt_err), (0,h-3), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)

    # if pred_D is not None:
    #     cv2.putText(pred_D,"Den Predict", (0,20), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)
    #     cv2.putText(pred_D,str(pred_cnt), (0,h-3), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)

    query_result = np.hstack((query_img,pred_D,attn))

    cv2.imwrite(os.path.join(output_dir,'{}.jpg'.format(fname)), query_result)


def save_density_map(query_img, pred_D,attn, GT_D,output_dir, fname='results.png', class_chosen=None, pred_cnt=None):

    if query_img is not None:
        _,h,w = query_img.shape
        query_img = query_img.cpu().numpy()
        query_img = 255.0 * (query_img - np.min(query_img) + 1e-10) / (1e-10 + np.max(query_img) - np.min(query_img))
        query_img = query_img.squeeze()
        query_img = query_img.transpose(1,2,0)
        query_img = cv2.cvtColor(query_img,cv2.COLOR_BGR2RGB)
    
    if pred_D is not None:
        pred_D = pred_D.cpu().detach().numpy()
        if pred_cnt is None:
            pred_cnt = np.sum(pred_D)
        pred_D = 255.0 * (pred_D - np.min(pred_D) + 1e-10) / (1e-10 + np.max(pred_D) - np.min(pred_D))
        pred_D = pred_D.squeeze()
        pred_D = cv2.applyColorMap(pred_D[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)
    
    if attn is not None:
        attn = attn.cpu().detach().numpy()
        attn = 255.0 * (attn - np.min(attn) + 1e-10) / (1e-10 + np.max(attn) - np.min(attn))
        attn = attn.squeeze()
        attn = cv2.applyColorMap(attn[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)

    if GT_D is not None:
        GT_D = GT_D.cpu().detach().numpy()
        gt_cnt = np.sum(GT_D)
        GT_D = 255.0 * (GT_D - np.min(GT_D) + 1e-10) / (1e-10 + np.max(GT_D) - np.min(GT_D))
        GT_D = GT_D.squeeze()
        GT_D = cv2.applyColorMap(GT_D[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)    

    
    # h,w = pred_D.shape[:2]
    # cv2.putText(query_img,class_chosen,(0,20),cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,0), 1)
    # if pred_D is not None:
    #     cv2.putText(pred_D,"Den Predict", (0,20), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)
    #     cv2.putText(pred_D,str(pred_cnt), (0,h-3), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)
    # if GT_D is not None:
    #     cv2.putText(GT_D,"Den GT", (0,20), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)
    #     cv2.putText(GT_D,str(gt_cnt), (0,h-3), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)

    query_result = np.hstack((query_img,pred_D,attn,GT_D))

    cv2.imwrite(os.path.join(output_dir,'{}.jpg'.format(fname)), query_result)



def get_model_dir(args: argparse.Namespace) -> str:
    """
    Obtain the directory to save/load the model
    """
    path = os.path.join(
        args.MODEL.model_dir,
        args.DATA.train_name,
        f'exp_{args.exp}'
    )
    return path

def get_model_dir_carpk(args: argparse.Namespace) -> str:
    """
    Obtain the directory to save/load the model
    """
    path = os.path.join(
        args.MODEL.model_dir,
        args.DATA.train_name,
        f'exp_{args.exp}',
        'inference_carpk'
    )
    return path

def get_model_dir_pucpr(args: argparse.Namespace) -> str:
    """
    Obtain the directory to save/load the model
    """
    path = os.path.join(
        args.MODEL.model_dir,
        args.DATA.train_name,
        f'exp_{args.exp}',
        'inference_pucpr'
    )
    return path