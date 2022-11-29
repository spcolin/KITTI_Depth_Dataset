import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import random

class KITTI_Depth(Dataset):
    def __init__(self,data_path,gt_path,file_path,target_height=352,target_width=1120,
                mode='train',random_crop=True,random_rotate=True,
                rotate_degree=1.0,depth_scale=256.0,random_flip=True,color_aug=True,use_right=True,do_kb_crop=True):

        self.data_path=data_path
        self.gt_path=gt_path

        self.target_height=target_height
        self.target_width=target_width
        self.mode=mode
        self.random_crop=random_crop
        self.random_rotate=random_rotate
        self.rotate_degree=rotate_degree
        self.depth_scale=depth_scale
        self.random_flip=random_flip
        self.color_aug=color_aug
        self.use_right=use_right
        self.do_kb_crop=do_kb_crop
        
        file=open(file_path)

        self.files=file.readlines()

        file.close()

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):

        sample_path = self.files[idx]

        if self.mode=='train':

            rgb_file = sample_path.split()[0]
            # depth_file = os.path.join(sample_path.split()[0].split('/')[0], sample_path.split()[1])
            depth_file = sample_path.split()[1]

            if self.use_right and (random.random() > 0.5):
                rgb_file.replace('image_02', 'image_03')
                depth_file.replace('image_02', 'image_03')
                
            image_path = os.path.join(self.data_path, rgb_file)
            depth_path = os.path.join(self.gt_path, depth_file)
        
            rgb = Image.open(image_path)
            depth = Image.open(depth_path)

            if self.do_kb_crop:
                height = rgb.height
                width = rgb.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth = depth.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                rgb = rgb.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            
            if self.random_crop:
                rgb,depth=self.crop_resize(rgb,depth,self.target_height,self.target_width)

            if self.random_rotate:
                random_angle = (random.random() - 0.5) * 2 * self.rotate_degree
                rgb = self.rotate_image(rgb, random_angle)
                depth = self.rotate_image(depth, random_angle, flag=Image.NEAREST)

            if self.random_flip:
                rgb,depth=self.flip_image(rgb,depth)

            if self.color_aug:
                rgb=self.augment_image(rgb)

            to_tensor_and_norm=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            rgb_tensor=to_tensor_and_norm(rgb)
            depth_tensor=transforms.ToTensor()(depth)

            depth_tensor=depth_tensor/self.depth_scale

            sample = {'image': rgb_tensor, 'depth': depth_tensor}

            return sample

        else:

            image_path = os.path.join(self.data_path,sample_path.split()[0])

            depth_path = os.path.join(self.gt_path,sample_path.split()[1])
            
            has_valid_depth = False

            try:
                depth = Image.open(depth_path)
                has_valid_depth = True
            except IOError:
                depth = False

            rgb=Image.open(image_path)

            if self.do_kb_crop:
                height = rgb.height
                width = rgb.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth = depth.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                rgb = rgb.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            to_tensor_and_norm=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            rgb_tensor=to_tensor_and_norm(rgb)
            depth_tensor=transforms.ToTensor()(depth)

            depth_tensor=depth_tensor/self.depth_scale

            sample = {'image': rgb_tensor, 'depth': depth_tensor,'has_valid_depth': has_valid_depth}

            return sample
            


    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def flip_image(self,rgb,depth):

        do_flip = random.random()
        if do_flip > 0.5:
            # print('flip image')
            rgb=rgb.transpose(Image.FLIP_LEFT_RIGHT)
            depth=depth.transpose(Image.FLIP_LEFT_RIGHT)

        return rgb,depth

    def crop_resize(self, rgb, depth, target_height, target_width):

        full_width,full_height=rgb.size

        height_scale=random.uniform(0.8,1)
        width_scale=random.uniform(0.8,1)

        height=int(full_height*height_scale)
        width=int(full_width*width_scale)

        x = random.randint(0, full_width - width)
        y = random.randint(0, full_height - height)

        rgb=rgb.crop((x,y,x+width,y+height))
        depth=depth.crop((x,y,x+width,y+height))

        rgb=rgb.resize((target_width,target_height),Image.ANTIALIAS)
        depth=depth.resize((target_width,target_height),Image.ANTIALIAS)

        return rgb,depth

    def augment_image(self, rgb):
        
        color_transform=transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1)

        rgb=color_transform(rgb)

        return rgb




