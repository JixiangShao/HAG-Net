import os
import time
import datetime
import torch
# import sys
# sys.path.append(r"D:\Codes\Deep learning\unet\save_weights")
from src import UNet,ResNetUNet,VGG16UNet,MobileV3Unet,MMUNet,R2U_Net,SegNet,NestedUNet,get_fcn_model,create_ce_net,UNet_HLAEM,UNet_HLAEM_GNN
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset

import transforms as T

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 根据输入的基础尺寸计算随机调整图像大小的最小和最大尺寸
        min_size = int(0.5 * base_size)  # 最小尺寸为基础尺寸的50%
        max_size = int(1.2 * base_size)  # 最大尺寸为基础尺寸的120%
        # 构建数据增强的变换序列，首先是随机调整图像大小
        trans = [T.RandomResize(min_size, max_size)]
        # 如果水平翻转概率大于0，则添加随机水平翻转的操作
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        # 如果垂直翻转概率大于0，则添加随机垂直翻转的操作
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.append(T.RandomContrast(contrast_factor=(0.8, 1.2)))
        trans.append(T.RandomGamma(gamma_limit=(0.7, 1.5)))
        # 在变换序列中添加随机裁剪、张量转换和归一化的操作
        trans.extend([
            T.RandomCrop(crop_size),  # 随机裁剪图像到指定大小
            T.ToTensor(),  # 将图像从PIL格式转换为张量格式
            T.Normalize(mean=mean, std=std),  # 对图像进行归一化处理
        ])
        # 将所有的数据增强操作组合成一个变换序列
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        # 调用时对输入的图像和目标（如标签）应用变换
        return self.transforms(img, target)

class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 定义评估（验证/测试）模式下的变换序列
        self.transforms = T.Compose([
            T.ToTensor(),  # 将图像从PIL格式转换为张量格式
            T.Normalize(mean=mean, std=std),  # 对图像进行归一化处理
        ])

    def __call__(self, img, target):
        # 调用时对输入的图像和目标（如标签）应用变换
        return self.transforms(img, target)

def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565  # 定义基础图像尺寸
    crop_size = 480  # 定义裁剪后的图像尺寸
    transforms = []
    if train:
        # 如果是训练模式，返回训练模式下的数据增强配置
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        # 如果是评估模式，返回评估模式下的变换配置
        return SegmentationPresetEval(mean=mean, std=std)





def create_model(num_classes):
    # 创建一个 UNet 模型实例，设置输入通道为 3（RGB图像），输出类别数为 num_classes，基础通道数为 32
    # model = UNet(in_channels=3, num_classes=num_classes)
    # model = ResNetUNet(num_classes=num_classes)
    # model = VGG16UNet(num_classes=2, pretrain_backbone=False)
    # model = MobileV3Unet(num_classes=2)
    # model = MMUNet(in_channels=3, num_classes=2)
    # model = R2U_Net(in_channels=3,num_classes=num_classes,t=2)
    # model = SegNet(in_channels=3,num_classes=num_classes)
    # model = NestedUNet(in_channels=3,num_classes=num_classes,deepsupervision=False)
    # model = get_fcn_model(model_type='fcn8s', num_classes=num_classes, backbone='vgg16', pretrained=True)
    # model = get_fcn_model(model_type='fcn32s', num_classes=num_classes, backbone='vgg16', pretrained=True)
    # model = create_ce_net(model_type="base",num_classes=num_classes,pretrained=True)
    # model = UNet_HLAEM(in_channels=3, num_classes=2)
    model = UNet_HLAEM_GNN(in_channels=3,num_classes=2)
    return model

def main(args):
    # 获取设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 批次大小
    batch_size = args.batch_size
    # 分割类别数（包括背景）
    num_classes = args.num_classes + 1

    # 图像均值和标准差
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 用于保存训练和验证信息的文件
    results_file = "500e_DRIVE{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 创建训练和测试数据集
    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = 0 if os.name == 'nt' else min([os.cpu_count() // 2, 8])  # 计算可用的 worker 数量，限制在最小的工作进程数和一些条件下的最小值
    train_loader = torch.utils.data.DataLoader(train_dataset,  # 创建训练数据加载器
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset, # 创建验证数据加载器
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)  # 创建模型实例
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad] # 获取需要优化的参数
    # 创建优化器

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,  weight_decay=args.weight_decay
    )
    # 创建混合精度训练的梯度缩放器（如果开启了混合精度训练）
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次（不是每个epoch）
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
     # 如果设置了恢复训练
    if args.resume:
        # 加载之前保存的模型状态
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        # 如果开启了混合精度训练，还需恢复梯度缩放器状态
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    # 初始化最佳 Dice 分数和开始时间
    best_dice = 0.
    start_time = time.time()
    print("准备进入训练循环...")  # 新增
    # 在训练脚本中修改评估调用（安全版本）
    for epoch in range(args.start_epoch, args.epochs):
        # 训练一个 epoch
        print(f"进入第{epoch}个epoch...")
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        try:
            # 尝试使用新的评估函数
            confmat, dice, extra_metrics = evaluate(model, val_loader, device=device, num_classes=num_classes)

            val_info = str(confmat)
            print(val_info)
            print(f"dice coefficient: {dice:.4f}")

            # 安全地访问 extra_metrics
            if extra_metrics is not None:
                print(
                    f"ACC: {extra_metrics.get('acc', 0):.4f} | SE: {extra_metrics.get('se', 0):.4f} | SP: {extra_metrics.get('sp', 0):.4f}")
                print(f"F1: {extra_metrics.get('f1', 0):.4f} | AUC: {extra_metrics.get('auc', 0):.4f}")
            else:
                print("额外指标计算失败，使用默认值")
                extra_metrics = {'acc': 0, 'se': 0, 'sp': 0, 'f1': 0, 'auc': 0}

        except Exception as e:
            print(f"评估过程中出现错误: {e}")
            # 回退到原来的评估方式
            confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
            val_info = str(confmat)
            print(val_info)
            print(f"dice coefficient: {dice:.4f}")
            extra_metrics = {'acc': 0, 'se': 0, 'sp': 0, 'f1': 0, 'auc': 0}

        # 将结果写入到文件中
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.4f}\n" \
                         f"ACC: {extra_metrics['acc']:.4f}\n" \
                         f"SE: {extra_metrics['se']:.4f}\n" \
                         f"SP: {extra_metrics['sp']:.4f}\n" \
                         f"F1: {extra_metrics['f1']:.4f}\n" \
                         f"AUC: {extra_metrics['auc']:.4f}\n"
            f.write(train_info + val_info + "\n\n")
        # 如果开启了保存最佳模型
        if args.save_best is True:
            # 如果当前 Dice 值优于历史最佳，则更新最佳 Dice 值
            if best_dice < dice:
                best_dice = dice
            else:
                continue
        # 准备要保存的模型状态
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        # 如果开启了混合精度训练，还需保存梯度缩放器的状态
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        # 根据条件选择保存最佳模型或每个 epoch 的模型
        if args.save_best is True:
            torch.save(save_file, "save_weights/UNet_HLAEM_GNN_best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
    # 计算总训练时间并打印
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=500, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0009, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # 混合精度训练参数
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # 如果保存模型的文件夹不存在，则创建它
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    # 执行主程序入口函数
    main(args)
