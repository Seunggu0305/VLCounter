import cv2
import os
import numpy as np
import random
import argparse
from typing import List
import json

import inflect
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import scipy.ndimage as ndimage

import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from .tokenizer import tokenize

TTensor = transforms.Compose([
    transforms.ToTensor()
])


Normalize = transforms.Compose([
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
])

Augmentation = transforms.Compose([
    transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15),
    transforms.GaussianBlur(kernel_size=(7, 9))
])


class ResizeTrainImage(object):
    """
    Resize the image so that:
        1. Image is equal to 384 * 384
        2. The new height and new width are divisible by 16
        3. The aspect ratio is possibly preserved
    Density map is cropped to have the same size(and position) with the cropped image
    Exemplar boxes may be outside the cropped area.
    Augmentation including Gaussian noise, Color jitter, Gaussian blur, Random affine, Random horizontal flip and Mosaic (or Random Crop if no Mosaic) is used.
    """

    # def __init__(self, data_path=Path('/workspace/FSC147/'), MAX_HW=384):
    #     super().__init__(data_path)
    def __init__(self, MAX_HW=384):
        super().__init__()
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, density, dots, im_id = sample['image'], sample['gt_density'], \
            sample['dots'], sample['id']
        W, H = image.size

        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        scale_factor = float(new_W) / W
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))

        # Augmentation probability
        aug_p = random.random()
        aug_flag = 0
        mosaic_flag = 0
        if aug_p < 0.4:  # 0.4
            aug_flag = 1
            # if aug_p < 0.25:  # 0.25
            #     aug_flag = 0
            #     mosaic_flag = 1

        # Gaussian noise
        resized_image = TTensor(resized_image)
        if aug_flag == 1:
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            re_image = resized_image + noise
            re_image = torch.clamp(re_image, 0, 1)

        # Color jitter and Gaussian blur
        if aug_flag == 1:
            re_image = Augmentation(re_image)

        # Random affine
        if aug_flag == 1:
            re1_image = re_image.transpose(0, 1).transpose(1, 2).numpy()
            keypoints = []
            for i in range(dots.shape[0]):
                keypoints.append(Keypoint(x=min(new_W - 1, int(dots[i][0] * scale_factor)), y=min(new_H - 1, int(dots[i][1]))))
            kps = KeypointsOnImage(keypoints, re1_image.shape)

            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-15, 15),
                    scale=(0.8, 1.2),
                    shear=(-10, 10),
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
                )
            ])
            re1_image, kps_aug = seq(image=re1_image, keypoints=kps)

            # Produce dot annotation map
            resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]), dtype='float32')
            for i in range(len(kps.keypoints)):
                if (int(kps_aug.keypoints[i].y) <= new_H - 1 and int(kps_aug.keypoints[i].x) <= new_W - 1) and not \
                        kps_aug.keypoints[i].is_out_of_image(re1_image):
                    resized_density[int(kps_aug.keypoints[i].y)][int(kps_aug.keypoints[i].x)] = 1
            resized_density = torch.from_numpy(resized_density)

            re_image = TTensor(re1_image)

        # Random horizontal flip
        if aug_flag == 1:
            flip_p = random.random()
            if flip_p > 0.5:
                re_image = TF.hflip(re_image)
                resized_density = TF.hflip(resized_density)

        # Random 384*384 crop in a new_W*384 image and 384*new_W density map
        if mosaic_flag == 0:
            if aug_flag == 0:
                re_image = resized_image
                resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]), dtype='float32')
                for i in range(dots.shape[0]):
                    resized_density[min(new_H - 1, int(dots[i][1]))][min(new_W - 1, int(dots[i][0] * scale_factor))] = 1
                resized_density = torch.from_numpy(resized_density)

            start = random.randint(0, new_W - 384)
            reresized_image = TF.crop(re_image, 0, start, 384, 384)
            reresized_density = resized_density[:, start:start + 384]
            reresized_image = Normalize(reresized_image)

        # Gaussian distribution density map
        reresized_density = ndimage.gaussian_filter(reresized_density.numpy(), sigma=(3,3), order=0)

        # Density map scale up
        reresized_density = reresized_density * 60
        reresized_density = torch.from_numpy(reresized_density).unsqueeze(0)


        return reresized_image, reresized_density


class MainTransform(object):
    def __init__(self, img_size):
        self.img_size = img_size
        self.img_trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
                        ])
    def __call__(self, sample):
        img, target, dots = sample['image'], sample['gt_density'], sample['dots']
        W, H = img.size
        new_H = 384
        new_W = 16 * int((W / H * 384) / 16)
        scale_factor_W = float(new_W) / W
        scale_factor_H = float(new_H) / H
        resized_img = transforms.Resize((new_H, new_W))(img)
        resized_img = self.img_trans(resized_img)
        
        gt_map = np.zeros((resized_img.shape[1], resized_img.shape[2]), dtype='float32')
        for i in range(dots.shape[0]):
            gt_map[min(new_H - 1, int(dots[i][1] * scale_factor_H))][min(new_W - 1, int(dots[i][0] * scale_factor_W))] = 1
        gt_map = ndimage.gaussian_filter(gt_map, sigma=(1, 1), order=0)
        gt_map = torch.from_numpy(gt_map)
        gt_map = gt_map



        # resized_target = cv2.resize(target,(self.img_size,self.img_size))
        # orig_count = np.sum(target)
        # new_count = np.sum(resized_target)
        # if new_count > 0: resized_target = resized_target * (orig_count/new_count)
    

        # resized_img = self.img_trans(resized_img)
        # resized_target = torch.from_numpy(resized_target).unsqueeze(0)
        return resized_img, gt_map

def get_train_loader(args: argparse.Namespace,mode: str) -> torch.utils.data.DataLoader:
    """
        Build the train loader. This is a episodic loader.
    """
    main_transform = ResizeTrainImage(MAX_HW=args.DATA.img_size)
    
    # ====== Build loader ======
    train_data = listDataset(
        args.DATA.data_root,
        prompt=args.prompt,
        singular=args.singular,
        mode = mode,
        main_transform = main_transform, 
    ) # mode = 'train'
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.TRAIN.batch_size,
        shuffle=True,
        # collate_fn=CutMixCollator(),
        num_workers=args.DATA.workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader


def get_val_loader(args: argparse.Namespace,mode: str,task=None) -> torch.utils.data.DataLoader:
    """
        Build the episodic validation loader.
    """
    main_transform = MainTransform(img_size=args.DATA.img_size)

    # ====== Build loader ======
    val_data = listDataset(
        args.DATA.data_root, 
        prompt=args.prompt,
        singular=args.singular,
        mode = mode, 
        main_transform = main_transform, 
    ) # mode = 'val' or 'test'

    val_loader = torch.utils.data.DataLoader(
        val_data,
        # batch_size=args.TRAIN.batch_size,
        batch_size=1,
        shuffle=False,
        num_workers=args.DATA.workers,
        pin_memory=True,
        drop_last=False,
    )

    return val_loader

class listDataset(Dataset):
    def __init__(self, data_root, prompt, singular, mode, main_transform):
        
        self.main_transform = main_transform
        self.data_root = data_root
        self.prompt = prompt
        self.singular = singular
        self.mode = mode
        self.train_class = ['shoes', 'roof tiles', 'supermarket shelf', 'pigeons', 'polka dot tiles', 'rice bags', 'straws', 'kidney beans', 'bananas', 'pens', 'meat skewers', 'bread rolls', 'coffee beans', 'swans', 'matches', 'oranges', 'caps', 'boxes', 'watermelon', 'beads', 'potatoes', 'calamari rings', 'biscuits', 'stapler pins', 'go game', 'chewing gum pieces', 'clams', 'people', 'bowls', 'crows', 'ice cream', 'cupcakes', 'lipstick', 'penguins', 'crayons', 'cars', 'nails', 'instant noodles', 'pearls', 'kitchen towels', 'goldfish snack', 'buffaloes', 'bricks', 'jeans', 'geese', 'cups', 'goats', 'coins', 'cotton balls', 'bees', 'peppers', 'baguette rolls', 'lighters', 'cartridges', 'spring rolls', 'stairs', 'balls', 'gemstones', 'alcohol bottles', 'onion rings', 'cement bags', 'cereals', 'cupcake tray', 'chopstick', 'cans', 'cows', 'screws', 'naan bread', 'bottles', 'cranes', 'm&m pieces', 'nuts', 'birthday candles', 'macarons', 'buns', 'mosaic tiles', 'boats', 'plates', 'tomatoes', 'zebras', 'cassettes', 'fishes', 'croissants', 'pencils', 'mini blinds', 'windows', 'candles', 'spoon', 'jade stones']

        anno_path = os.path.join(self.data_root, "annotation_FSC147_384.json")
        with open(anno_path) as f:
            self.annotations = json.load(f)
        
        data_split_file = os.path.join(self.data_root, "Train_Test_Val_FSC_147.json")
        with open(data_split_file) as f:
            data_split = json.load(f)
        img_ids = data_split[mode]

        image_classes_file = os.path.join(self.data_root, 'ImageClasses_FSC147.txt')
        img_classes = {}
        with open(image_classes_file) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                im_id, class_name = line.strip().split('\t')
                img_classes[im_id] = class_name
                
        self.data_list = [(img_classes[img_id],img_id) for img_id in img_ids]
        self.nSamples = len(self.data_list)
        self.single_plural_prompt_templates = ['A photo of a {}.',
                                'A photo of a small {}.',
                                'A photo of a medium {}.',
                                'A photo of a large {}.',
                                'This is a photo of a {}.',
                                'This is a photo of a small {}.',
                                'This is a photo of a medium {}.',
                                'This is a photo of a large {}.',
                                'A {} in the scene.',
                                'A photo of a {} in the scene.',
                                'There is a {} in the scene.',
                                'There is the {} in the scene.',
                                'This is a {} in the scene.',
                                'This is the {} in the scene.',
                                'This is one {} in the scene.',
                            ]
        self.multi_plural_prompt_templates = ['a photo of a number of {}.',
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
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        class_chosen, img_id = self.data_list[index]
        list_path = [self.data_root, 'image', img_id]
        query_img_path = os.path.join(*list_path)

        anno = self.annotations[img_id]
        dots = np.array(anno['points'])

        image = Image.open(query_img_path)
        image.load()
        gt_path = query_img_path.replace('.jpg','.npy').replace('image','gt')
        density = np.load(gt_path).astype(np.float32)
        # m_flag = 0
        
        sample = {'image': image, 'gt_density': density, 'dots': dots, 'id': img_id}
                #   'm_flag': m_flag}
        
        query_img, query_density = self.main_transform(sample)
        
        engine = inflect.engine()
        if self.prompt == "plural":
            text = [template.format(engine.plural(class_chosen)) if engine.singular_noun(class_chosen) == False else template.format(class_chosen) for template in self.multi_plural_prompt_templates]
        elif self.prompt == "single":
            text = [template.format(class_chosen) if engine.singular_noun(class_chosen) == False else template.format(engine.plural(class_chosen)) for template in self.single_plural_prompt_templates]
        else:
            raise NotImplementedError
        tokenized_text = tokenize(text)
        
        return query_img, query_density, tokenized_text, class_chosen