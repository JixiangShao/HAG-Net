import os
import time
import datetime
import torch
from src import UNet, ResNetUNet, R2U_Net, SegNet, NestedUNet, get_fcn_model, create_ce_net, HAG_Net
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset

import transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # Calculate minimum and maximum random resizing sizes based on base input size
        min_size = int(0.5 * base_size)  # Minimum size equals 50% of base size
        max_size = int(1.2 * base_size)  # Maximum size equals 120% of base size
        # Build augmentation transform list, starting with random resizing
        trans = [T.RandomResize(min_size, max_size)]
        # Add random horizontal flip if flip probability > 0
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        # Add random vertical flip if flip probability > 0
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.append(T.RandomContrast(contrast_factor=(0.8, 1.2)))
        trans.append(T.RandomGamma(gamma_limit=(0.7, 1.5)))
        # Append random cropping, tensor conversion and normalization to transform list
        trans.extend([
            T.RandomCrop(crop_size),  # Random crop image to target size
            T.ToTensor(),  # Convert PIL image to tensor format
            T.Normalize(mean=mean, std=std),  # Perform image normalization
        ])
        # Combine all data augmentation operations into a single transform pipeline
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        # Apply all transforms to input image and segmentation label during call
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # Define transform pipeline for validation/test evaluation mode
        self.transforms = T.Compose([
            T.ToTensor(),  # Convert PIL image to tensor format
            T.Normalize(mean=mean, std=std),  # Perform image normalization
        ])

    def __call__(self, img, target):
        # Apply evaluation transforms to input image and segmentation label during call
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565  # Define base input image resolution
    crop_size = 480  # Define target cropped image resolution
    transforms = []
    if train:
        # Return training augmentation pipeline if in training mode
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        # Return evaluation transform pipeline if in validation/test mode
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    # Initialize UNet model, input channels=3 for RGB images, output classes=num_classes, base channel=32
    # model = UNet(in_channels=3, num_classes=num_classes)
    # model = ResNetUNet(num_classes=num_classes)
    # model = R2U_Net(in_channels=3,num_classes=num_classes,t=2)
    # model = SegNet(in_channels=3,num_classes=num_classes)
    # model = NestedUNet(in_channels=3,num_classes=num_classes,deepsupervision=False)
    # model = get_fcn_model(model_type='fcn8s', num_classes=num_classes, backbone='vgg16', pretrained=True)
    # model = get_fcn_model(model_type='fcn32s', num_classes=num_classes, backbone='vgg16', pretrained=True)
    # model = create_ce_net(model_type="base",num_classes=num_classes,pretrained=True)
    model = HAG_Net(in_channels=3, num_classes=2)
    return model


def main(args):
    # Get computation device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Training batch size
    batch_size = args.batch_size
    # Number of segmentation classes (including background)
    num_classes = args.num_classes + 1

    # Image normalization mean and std
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # File to record training and validation metrics
    results_file = "500e_DRIVE{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Create training and validation dataset instances
    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = 0 if os.name == 'nt' else min(
        [os.cpu_count() // 2, 8])  # Calculate available dataloader workers, cap maximum worker count
    train_loader = torch.utils.data.DataLoader(train_dataset,  # Initialize training dataloader
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,  # Initialize validation dataloader
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)  # Initialize segmentation model
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]  # Collect trainable model parameters
    # Initialize optimizer

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr, weight_decay=args.weight_decay
    )
    # Initialize gradient scaler for automatic mixed precision training (if AMP enabled)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Create learning rate scheduler, update learning rate per training step (not per epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    # Resume training from checkpoint if specified
    if args.resume:
        # Load saved model checkpoint file
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        # Restore gradient scaler state if mixed precision training is enabled
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    # Initialize best dice score and training start timestamp
    best_dice = 0.
    start_time = time.time()
    print("Start training loop...")  # New print log
    # Use safe evaluation function wrapper in training script
    for epoch in range(args.start_epoch, args.epochs):
        # Train model for one full epoch
        print(f"Starting epoch {epoch}...")
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        try:
            # Attempt to run enhanced evaluation function
            confmat, dice, extra_metrics = evaluate(model, val_loader, device=device, num_classes=num_classes)

            val_info = str(confmat)
            print(val_info)
            print(f"dice coefficient: {dice:.4f}")

            # Safely fetch additional evaluation metrics
            if extra_metrics is not None:
                print(
                    f"ACC: {extra_metrics.get('acc', 0):.4f} | SE: {extra_metrics.get('se', 0):.4f} | SP: {extra_metrics.get('sp', 0):.4f}")
                print(f"F1: {extra_metrics.get('f1', 0):.4f} | AUC: {extra_metrics.get('auc', 0):.4f}")
            else:
                print("Extra metrics calculation failed, fill with default values")
                extra_metrics = {'acc': 0, 'se': 0, 'sp': 0, 'f1': 0, 'auc': 0}

        except Exception as e:
            print(f"Error occurred during evaluation: {e}")
            # Fallback to original basic evaluation function
            confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
            val_info = str(confmat)
            print(val_info)
            print(f"dice coefficient: {dice:.4f}")
            extra_metrics = {'acc': 0, 'se': 0, 'sp': 0, 'f1': 0, 'auc': 0}

        # Write all metrics to log file
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
        # Enable saving only the best-performing model checkpoint
        if args.save_best is True:
            # Update best dice score if current dice surpasses historical maximum
            if best_dice < dice:
                best_dice = dice
            else:
                continue
        # Package all states to save checkpoint
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        # Save gradient scaler state if mixed precision training is enabled
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        # Save either best model only or checkpoint per epoch based on configuration
        if args.save_best is True:
            torch.save(save_file, "save_weights_DRIVE/HAG_Net_model.pth")
        else:
            torch.save(save_file, "save_weights_DRIVE/model_{}.pth".format(epoch))
    # Calculate total training time and print log
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE dataset root directory")

    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training computation device")
    parser.add_argument("-b", "--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=500, type=int, metavar="N",
                        help="total number of training epochs")

    parser.add_argument('--lr', default=0.0009, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='optimizer momentum coefficient')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay regularization (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='metric print interval per epoch')
    parser.add_argument('--resume', default='', help='checkpoint file path for resume training')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='starting epoch index')
    parser.add_argument('--save-best', default=True, type=bool, help='only save checkpoint with maximum dice score')
    # Mixed precision training argument
    parser.add_argument("--amp", default=False, type=bool,
                        help="enable automatic mixed precision training via torch.cuda.amp")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # Create weight save directory if it does not exist
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    # Execute main training function entry
    main(args)