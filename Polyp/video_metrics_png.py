import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data as data
from dataloaders.video_list import test_dataset
from eval.metrics import Smeasure, Emeasure, MAE, WeightedFmeasure, Fmeasure
from eval.metrics import Medical, MedicalFixedSimple, MedicalFixedSimpleImproved

from lib.unet import Unet
from lib.vivim import Vivim
from lib.segformer import SegFormer
from lib.pkechonet import PKEchoNet
from lib.transunet import VisionTransformer
from lib.configs.transunet import vit_seg_configs as configs
from lib.net import ResNet18_FSTMamba_BSTAFusion, SegFormer_FSTMamba_BSTAFusion


if __name__ == "__main__":
    # --- 测试参数 ---
    parser = argparse.ArgumentParser()
    
    # CVC-ClinicDB-612-Test/CVC-ClinicDB-612-Valid/CVC-ColonDB-300
    parser.add_argument('--dataset_root',  type=str, default='/root/data/VPS-dataset') 
    parser.add_argument('--testsplit', type=str, default='CVC-ClinicDB-612-Test', help='val dataset')
    
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')  # image size
    parser.add_argument('--seed', type=int, default=2021, help='random seed to use. Default=123')

    # model name
    parser.add_argument('--save_path', type=str,default='./result/Vivim/', help='the path to save model and log')
    parser.add_argument('--valonly', action='store_true', default=False, help='skip training during training')
    
    # TransUNet configuration
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')

    opt = parser.parse_args()

    # --- 载入数据 ---
    val_dataset = test_dataset(dataset_root=opt.dataset_root, split=opt.testsplit, testsize=opt.trainsize)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8)
    print('Val with %d image pairs' % len(val_loader))

    # --- 载入模型 ---
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
    # model = SegFormer_FSTMamba_BSTAFusion(in_chan=3, n_classes=1, mode='segformer')
    model.cuda()
    checkpoint = torch.load('/root/SALI/snapshot/Vivim/Vivim_epoch_33_612t.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    # --- 平均指标 ---
    sm, adpem, adpfm, mae = [], [], [], []
    dice, iou, spe, sen = [], [], [], []

    # --- 模型推理 ---
    for i, data_blob in enumerate(val_loader, start=1):
        images, gt, name, scene, case_idx = data_blob
        gt /= (gt.max() + 1e-8)
        images, gt = images.cuda(), gt.cuda()
        res, gt = model(images).squeeze(0), gt.squeeze(0)
        res = F.upsample(res, size=gt.shape[2:], mode='bilinear', align_corners=False)
        
        # --- 数据准备 ---
        # 移除 batch 和多余的通道维度: (1, 10, 1, h, w) -> (10, h, w)
        # 注意：metrics.py 内部会处理归一化等操作，所以这里只需要 numpy 数组
        pred_sequence_np = res.sigmoid().data.cpu().numpy().squeeze()
        gt_sequence_np = gt.cpu().numpy().squeeze()

        # 转换为符合Metric的输入数据
        pred_uint8 = (pred_sequence_np * 255).astype(np.uint8)
        if gt_sequence_np.max() <= 1.0:
            # 如果GT是 [0,1] 范围，先转换为 [0,255]
            gt_uint8 = (gt_sequence_np * 255).astype(np.uint8)
        else:
            # 如果已经是 [0,255] 范围，直接使用
            gt_uint8 = gt_sequence_np.astype(np.uint8)
        # 确保GT是严格二值的（只有0和255）
        gt_uint8[gt_uint8 > 128] = 255
        gt_uint8[gt_uint8 <= 128] = 0

        # 获取帧数、高度和宽度
        t, h, w = pred_uint8.shape
        print(f"处理视频序列{name}，共 {t} 帧，尺寸 {h}x{w}")

        # --- 初始化所有需要的评估器 ---
        # 传入 length 参数，告知评估器总共会有多少帧
        sm_evaluator = Smeasure(length=t)
        em_evaluator = Emeasure(length=t)
        mae_evaluator = MAE(length=t)
        wf_evaluator = WeightedFmeasure(length=t) # 可选
        fm_evaluator = Fmeasure(length=t)        # 可选
        # medical_evaluator = Medical(length=t)    # 可选，用于 Dice, IoU 等
        medical_evaluator = MedicalFixedSimpleImproved()

        # --- 逐帧评估 ---
        for frame_idx in range(t):
            # 获取当前帧的预测和真值 (都是 2D 数组，形状 h x w)
            pred_frame = pred_uint8[frame_idx, :, :]
            gt_frame = gt_uint8[frame_idx, :, :]

            # 对每个评估器调用 step 方法，传入当前帧和它的索引
            sm_evaluator.step(pred_frame, gt_frame, frame_idx)
            em_evaluator.step(pred_frame, gt_frame, frame_idx)
            mae_evaluator.step(pred_frame, gt_frame, frame_idx)
            wf_evaluator.step(pred_frame, gt_frame, frame_idx) # 可选
            fm_evaluator.step(pred_frame, gt_frame, frame_idx) # 可选
            medical_evaluator.step(pred_frame, gt_frame) # 可选

        # --- 获取最终结果 ---
        results = {}
        results.update(sm_evaluator.get_results())
        results.update(em_evaluator.get_results())
        results.update(mae_evaluator.get_results())
        results.update(wf_evaluator.get_results()) # 可选
        results.update(fm_evaluator.get_results()) # 可选
        results.update(medical_evaluator.get_results()) # 可选

        # --- 保存图片 ---
        save_path = os.path.join(opt.save_path, opt.testsplit)
        os.makedirs(save_path, exist_ok=True)
        pred_binary = np.zeros_like(pred_uint8)
        pred_binary[pred_uint8 > 128] = 255  # 阈值0.5对应128
        for frame_idx in range(t):
            pred_frame = pred_binary[frame_idx, :, :]
            pred_img = Image.fromarray(pred_frame, mode='L')
            pred_img.save(os.path.join(save_path, f'{name[frame_idx][0].split('.')[0]}.png'))

        # --- 打印结果 ---
        # print("\n===== 视频序列评估结果 =====")
        for metric_name, value in results.items():
            # 处理可能为数组的指标 (如 meanEm 是256个阈值下的曲线)
            # if isinstance(value, np.ndarray):
            #     print(f"{metric_name}: 形状 {value.shape}, 均值 {value.mean():.4f}")
            # else:
            #     print(f"{metric_name}: {value:.4f}")
            # --- 保存结果 ---
            if metric_name == "Smeasure":
                sm.append(value.mean())
            elif metric_name == "adpEm":
                adpem.append(value.mean())
            elif metric_name == "adpFm":
                adpfm.append(value.mean())
            elif metric_name == "MAE":
                mae.append(value.mean())
            elif metric_name == "Dice":
                dice.append(value.mean())
            elif metric_name == "IoU":
                iou.append(value.mean())
            elif metric_name == "Sensitivity":
                sen.append(value.mean())
            elif metric_name == "Specificity":
                spe.append(value.mean())
    print(dice)
    print("\n===== 视频集合评估结果 =====")
    print(f"smeasure: {np.mean(sm):.4f}, adpem: {np.mean(adpem):.4f},\
           adpfm: {np.mean(adpfm):.4f}, mae: {np.mean(mae):.4f},\
              dice: {np.mean(dice):.4f}, iou: {np.mean(iou):.4f}, \
                sen: {np.mean(sen):.4f}, spe: {np.mean(spe):.4f}")
            