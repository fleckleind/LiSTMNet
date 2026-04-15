import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import time
from glob import glob
import os.path as osp
import pdb
import torch


# several data augumentation strategies
def cv_random_flip(imgs, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(len(label)):
            label[i] = label[i].transpose(Image.FLIP_LEFT_RIGHT)
       
    return imgs, label

def randomCrop(imgs, label):
    border = 30
    image_width = imgs[0].size[0]
    image_height = imgs[0].size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    
    for i in range(len(imgs)):
        imgs[i] = imgs[i].crop(random_region)
    for i in range(len(label)):
        label[i] = label[i].crop(random_region)

    return imgs, label

def randomRotation(imgs, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        for i in range(len(imgs)):
            imgs[i] = imgs[i].rotate(random_angle, mode)

        for i in range(len(label)):
            label[i] = label[i].rotate(random_angle, mode)

    return imgs, label

def colorEnhance(imgs):
    for i in range(len(imgs)):
        bright_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Brightness(imgs[i]).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Contrast(imgs[i]).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        imgs[i] = ImageEnhance.Color(imgs[i]).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        imgs[i] = ImageEnhance.Sharpness(imgs[i]).enhance(sharp_intensity)
    return imgs

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255       
    return Image.fromarray(img)

class VideoDataset(data.Dataset):
    def __init__(self, dataset_root, trainsize=352, batchsize=8, 
                 split='MoCA-Video-Train', clip_len=10, stride=9):
        """
        Args:
            dataset_root: 数据根目录
            trainsize: 图像大小
            batchsize: 批次大小
            split: 数据集划分
            clip_len: 连续帧的长度（默认为5）
            stride: 滑动窗口的步长（默认为1）
        """
        self.trainsize = trainsize
        self.clip_len = clip_len  # 连续帧数
        self.stride = stride      # 滑动步长
        self.image_list = []      # 存储每个clip的图片路径
        self.gt_list = []         # 存储每个clip的GT路径
        self.extra_info = []
        self.case_idx = []
        
        img_root = '{}/{}/Frame'.format(dataset_root, split)
        
        for case in os.listdir(osp.join(img_root)):
            images = sorted(glob(osp.join(img_root, case, '*.jpg')))
            gt_list = sorted(glob(osp.join(img_root.replace('Frame', 'GT'), case, '*.png')))
            
            assert len(images) == len(gt_list), \
                f"Case {case}: images({len(images)}) != gts({len(gt_list)})"
            
            # 生成连续帧clips
            for start_idx in range(0, len(images) - self.clip_len + 1, self.stride):
                # 提取连续clip_len帧
                clip_indices = list(range(start_idx, start_idx + self.clip_len))
                
                # 收集clip的路径
                clip_images = [images[idx] for idx in clip_indices]
                clip_gts = [gt_list[idx] for idx in clip_indices]
                
                self.image_list.append(clip_images)  # [clip_len, 图片路径]
                self.gt_list.append(clip_gts)        # [clip_len, GT路径]
                self.extra_info.append((case, start_idx))
                self.case_idx.append(start_idx)
        
        print(f"加载了 {len(self.image_list)} 个视频clips，每个clip {self.clip_len} 帧")
        
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        """
        返回:
            imgs: [clip_len, 3, H, W] 的图片张量
            gts: [clip_len, 1, H, W] 的GT张量
            case_idx: 案例索引
        """
        img_paths = self.image_list[index]  # [clip_len, 路径]
        gt_paths = self.gt_list[index]      # [clip_len, 路径]
        
        # 加载所有图片和GT
        imgs = []
        gts = []
        
        for img_path in img_paths:
            img = self.rgb_loader(img_path)
            imgs.append(img)
        
        for gt_path in gt_paths:
            gt = self.binary_loader(gt_path)
            gts.append(gt)
        
        # 数据增强（同时对整个clip进行）
        imgs, gts = cv_random_flip(imgs, gts)
        imgs, gts = randomRotation(imgs, gts)
        imgs = colorEnhance(imgs)  # 只增强图片
        
        # 转换
        img_tensors = []
        gt_tensors = []
        
        for i in range(len(imgs)):
            img_tensors.append(self.img_transform(imgs[i]))
        
        for i in range(len(gts)):
            gts[i] = randomPeper(gts[i])
            gt_tensors.append(self.gt_transform(gts[i]))
        
        imgs_stack = torch.stack(img_tensors, dim=0)  # [clip_len, 3, H, W]
        gts_stack = torch.stack(gt_tensors, dim=0)    # [clip_len, 1, H, W]
        
        return imgs_stack, gts_stack, index

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return len(self.image_list)

# dataloader for training
def get_loader(dataset_root, batchsize, trainsize, train_split,
               shuffle=True, num_workers=8, pin_memory=True,):

    dataset = VideoDataset(dataset_root, trainsize, batchsize, split=train_split)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset(data.Dataset):
    def __init__(self, dataset_root, split='TestHardDataset/Seen', testsize=352, clip_len=10):
        """
        Args:
            dataset_root: 数据根目录
            split: 数据集划分
            testsize: 测试图像大小
            clip_len: 连续帧的长度（默认为5）
        """
        self.testsize = testsize
        self.clip_len = clip_len
        self.image_list = []  # 存储每个视频的中间clip
        self.gt_list = []     # 存储每个视频的中间clip的GT
        self.extra_info = []
        self.case_idx = [] 

        img_root = '{}/VPS-TestSet/{}/Frame'.format(dataset_root, split)
        
        for case in os.listdir(osp.join(img_root)):
            images = sorted(glob(osp.join(img_root, case, '*.jpg')))
            gt_list = sorted(glob(osp.join(img_root.replace('Frame', 'GT'), case, '*.png')))
            
            assert len(images) == len(gt_list), f"Case {case}: images({len(images)}) != gts({len(gt_list)})"
            
            # 只处理帧数足够的视频
            if len(images) >= self.clip_len:
                # 计算中间clip的起始帧
                # 方法：找到最中间的clip
                total_frames = len(images)
                max_start = total_frames - self.clip_len
                
                # 取最中间的起始位置
                start_idx = max_start // 2
                
                # 提取中间clip
                clip_indices = list(range(start_idx, start_idx + self.clip_len))
                clip_images = [images[idx] for idx in clip_indices]
                clip_gts = [gt_list[idx] for idx in clip_indices]
                
                self.image_list.append(clip_images)  # [clip_len, 图片路径]
                self.gt_list.append(clip_gts)        # [clip_len, GT路径]
                self.extra_info.append((case, start_idx))
                self.case_idx.append(start_idx)
            else:
                print(f"警告: 跳过测试视频 {case}，帧数({len(images)}) < {self.clip_len}")

        # transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()

        self.size = len(self.image_list)
        print(f"加载了 {self.size} 个测试视频，每个取中间 {self.clip_len} 帧")

    def __getitem__(self, index):
        """
        返回:
            imgs: [clip_len, 3, H, W] 的图片张量
            gts: [clip_len, 1, H, W] 的GT张量（所有帧的GT）
            names: 图片名称列表
            scene: 场景名称
            case_idx: 案例索引
        """
        img_paths = self.image_list[index]
        gt_paths = self.gt_list[index]
        
        imgs = []
        gts = []
        names = []
        
        for img_path, gt_path in zip(img_paths, gt_paths):
            img = self.rgb_loader(img_path)
            gt = self.binary_loader(gt_path)
            
            imgs.append(self.transform(img))
            gts.append(self.gt_transform(gt))
            names.append(img_path.split('/')[-1])
        
        imgs_stack = torch.stack(imgs, dim=0)  # [clip_len, 3, H, W]
        gts_stack = torch.stack(gts, dim=0)    # [clip_len, 1, H, W]
        
        # 获取场景信息（根据实际路径结构调整）
        scene = img_paths[0].split('/')[-3]
        case_idx = index
        
        return imgs_stack, gts_stack, names, scene, case_idx

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
    def __len__(self):
        return self.size
    