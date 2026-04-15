from __future__ import print_function, division
import os
import logging
import sys
from tqdm import tqdm
import argparse

sys.path.append('dataloaders')

import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
# import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from lib import VideoModel as Network
from utils.utils import clip_gradient, adjust_lr
from dataloaders import video_dataloader,get_loader
from tensorboardX import SummaryWriter
from dataloaders.video_list import get_loader, test_dataset

from lib.unet import Unet
from lib.vivim import Vivim
from lib.segformer import SegFormer
from lib.pkechonet import PKEchoNet
from lib.transunet import VisionTransformer
from lib.configs.transunet import vit_seg_configs as configs
from lib.net import ResNet18_FSTMamba_BSTAFusion, SegFormer_FSTMamba_BSTAFusion


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def cofficent_calculate(preds,gts,threshold=0.5):
    eps = 1e-5
    preds = preds > threshold
    intersection = (preds * gts).sum()
    union =(preds + gts).sum()
    dice = 2 * intersection  / (union + eps)
    iou = intersection/(union - intersection + eps)
    return (dice, iou)

def freeze_network(model):
    for name, p in model.named_parameters():
        if "fusion_conv" not in name:
            p.requires_grad = False


#######################################################################################################################

def train(train_loader, model, optimizer, epoch, save_path, writer, freq):
    """
    train function
    """
    global step
    
    model.train()
    
    loss_all = 0
    epoch_step = 0
    
    try:
        for i, data_blob in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images, gt = data_blob[0].cuda(), data_blob[1].cuda()

            preds = model(images)
            loss = structure_loss(preds[:,0], gt[:,0]) + structure_loss(preds[:,1], gt[:,1])\
                + structure_loss(preds[:,2], gt[:,2]) + structure_loss(preds[:,3], gt[:,3])\
                + structure_loss(preds[:,4], gt[:,4]) + structure_loss(preds[:,5], gt[:,5])\
                + structure_loss(preds[:,6], gt[:,6]) + structure_loss(preds[:,7], gt[:,7])\
                + structure_loss(preds[:,8], gt[:,8]) + structure_loss(preds[:,9], gt[:,9]) 
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data
        
            if i % 50 == 0 or i == total_step or i == 1:
                print(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                     format(epoch, opt.epoch, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                     format(epoch, opt.epoch, i, total_step, loss.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics', {'Loss_total': loss.data}, global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0][0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train_RGB', grid_image, step)
                grid_image = make_grid(gt[0][0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train_GT', grid_image, step)
                # TensorboardX-Outputs
                res = preds[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('train_Pred_final', torch.tensor(res), step, dataformats='HW')
       
        loss_all /= epoch_step
        print('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        #if epoch % 50 == 0:
        torch.save(model.state_dict(), save_path + 'Vivim_epoch_{}.pth'.format(epoch))  # model name
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Vivim_epoch_{}.pth'.format(epoch))  # model name
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer, mem_freq):
    """
    validation function
    """
    global best_mae, best_epoch, best_dice, step_val
    model.eval()
    
    with torch.no_grad():
        meandice_case = {}
        mae_sum = 0
        dice_sum = 0
        for i, data_blob in tqdm(enumerate(test_loader)):
            images, gt, name, scene, case_idx = data_blob
            gt /= (gt.max() + 1e-8)
            images, gt = images.cuda(), gt.cuda()
            res, gt = model(images).squeeze(0), gt.squeeze(0)

            res = F.upsample(res, size=gt.shape[2:], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            gt = gt.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
            dice = cofficent_calculate(res,gt)[0]
            if meandice_case.get(scene) is None:
                    meandice_case[scene] = [] 
            meandice_case[scene].append(dice)
            dice_sum += dice               

        mae = mae_sum / len(test_loader)
        dice = dice_sum / len(test_loader)
        Meandice_onCase = [np.mean(meandice_case[scene]) for scene in meandice_case]
        Meandice_onCase = sum(Meandice_onCase)/len(Meandice_onCase)

        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        writer.add_scalar('DICE', torch.tensor(dice), global_step=epoch)
        writer.add_scalar('meanDICE_on_cases', torch.tensor(Meandice_onCase), global_step=epoch)

        if dice > best_dice:
            best_dice = dice
            best_epoch = epoch
        # torch.save(model.state_dict(), save_path + 'Vivim_epoch_{}_dice.pth'.format(epoch))  # model name
        # print('Save state_dict successfully! Best epoch:{}.'.format(best_epoch))
        logging.info('[Val Info]:Epoch:{}, DICE:{}'.format(epoch, dice))
        print('[Val Info]:Epoch:{}, DICE:{}'.format(epoch, dice))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--dataset_root',  type=str, default='/root/data/VPS-dataset') 
    parser.add_argument('--trainsplit', type=str, default='VPS-TrainSet', help='train dataset')
    # CVC-ClinicDB-612-Test/CVC-ClinicDB-612-Valid/CVC-ColonDB-300
    parser.add_argument('--testsplit', type=str, default='CVC-ClinicDB-612-Valid', help='val dataset')
    
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')

    parser.add_argument('--pretrained_weights', type=str, default='./pretrained/pvt_v2_b5.pth',
                        help='path to the pretrained model')
    
    parser.add_argument('--resume', type=str, default='', help='train from checkpoints')
    parser.add_argument('--seed', type=int, default=2021, help='random seed to use. Default=123')
    
    parser.add_argument('--mem_freq', type=int, default=5, help='mem every n frames')
    parser.add_argument('--test_mem_length', type=int, default=50, help='max num of memory frames')
    
    # model name
    parser.add_argument('--save_path', type=str,default='./snapshot/Vivim/', help='the path to save model and log')
    parser.add_argument('--valonly', action='store_true', default=False, help='skip training during training')
    
    # TransUNet configuration
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    
    opt = parser.parse_args()

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(filename=os.path.join(save_path,'train.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

    # build the model
    # model = Network(opt).cuda()
    # model = Unet(img_channel=3, num_classes=1)
    # model = SegFormer(num_classes=1)
    # config_vit = configs.get_r50_b16_config()
    # config_vit.n_classes, config_vit.n_skip = 1, 3
    # config_vit.patches.grid = (
    #     int(opt.trainsize / opt.vit_patches_size), 
    #     int(opt.trainsize / opt.vit_patches_size))
    # model = VisionTransformer(
    #     config_vit, img_size=opt.trainsize, num_classes=config_vit.n_classes)
    # model.load_from(weights=np.load(config_vit.pretrained_path))
    # model = PKEchoNet(in_channels=3, num_classes=1, num_frame=10)
    model = Vivim(in_chans=3, out_chans=1)
    # model = ResNet18_FSTMamba_BSTAFusion(in_chan=3, n_classes=1, backbone='resnet18')
    model.cuda()

    # pdb.set_trace()
    logging.info('save to {}'.format(save_path))
    logging.info("Network-Train")

    #freeze_network(model)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    # load data
    logging.info('===> Loading datasets')
    print('===> Loading datasets')

    train_loader, val_loader = video_dataloader(opt)
    total_step = len(train_loader)

    # logging
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.resume, save_path, opt.decay_epoch))
    
    step = 0
    step_val = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_dice = 0
    best_epoch = 0

    freq = opt.mem_freq // opt.batchsize * opt.batchsize
    skip_list = [10,15,20,25,5]
    skip_i = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        if not opt.valonly:
            train(train_loader, model, optimizer, epoch, save_path, writer, freq)
        if isinstance(model,torch.nn.DataParallel):
            model = model.module
        model.cuda(0)
        val(val_loader, model, epoch, save_path, writer, opt.mem_freq)
      
