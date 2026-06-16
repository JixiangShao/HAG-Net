import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target, boundary_loss, total_variation_loss
import numpy as np
from sklearn.metrics import roc_auc_score

def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        # Ignore pixels with value 255 in target, which represent object edges or padding regions
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        loss += 0.2 * boundary_loss(x, target)  # Adjustable weight coefficient
        if num_classes == 2:
            # Stable implementation of Focal Loss
            gamma = 2.0
            alpha = 0.25

            logpt = -nn.functional.cross_entropy(x, target, weight=loss_weight,
                                                 ignore_index=ignore_index, reduction='none')
            pt = torch.exp(logpt)
            focal_loss = -alpha * (1 - pt) ** gamma * logpt
            focal_loss = focal_loss.mean()

            loss = 0.7 * loss + 0.3 * focal_loss

        # Dice loss calculation
        if dice:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)

        # Total variation loss to enhance segmentation continuity
        loss += total_variation_loss(x, weight=5e-5)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # New variables for computing extra evaluation metrics
    total_acc, total_se, total_sp, total_f1, total_auc = 0.0, 0.0, 0.0, 0.0, 0.0
    total_samples = 0

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            # Get predicted probability for foreground class to calculate AUC
            pred_prob = torch.softmax(output, dim=1)[:, 1]  # Probability of foreground category
            pred = output.argmax(1)  # Predicted class labels

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

            # Calculate additional metrics for current batch
            batch_size = image.shape[0]
            for i in range(batch_size):
                # Convert tensors to numpy arrays for metric computation
                pred_np = pred[i].cpu().numpy().flatten()
                target_np = target[i].cpu().numpy().flatten()
                pred_prob_np = pred_prob[i].cpu().numpy().flatten()

                # Mask out ignored pixels if present
                valid_mask = target_np != 255
                if valid_mask.sum() == 0:
                    continue

                pred_valid = pred_np[valid_mask]
                target_valid = target_np[valid_mask]
                pred_prob_valid = pred_prob_np[valid_mask]

                # Compute elements of confusion matrix
                tp = np.sum((pred_valid == 1) & (target_valid == 1))
                tn = np.sum((pred_valid == 0) & (target_valid == 0))
                fp = np.sum((pred_valid == 1) & (target_valid == 0))
                fn = np.sum((pred_valid == 0) & (target_valid == 1))

                # Calculate evaluation metrics
                acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
                se = tp / (tp + fn + 1e-6)  # Sensitivity (Recall)
                sp = tn / (tn + fp + 1e-6)  # Specificity
                precision = tp / (tp + fp + 1e-6)
                f1 = 2 * (precision * se) / (precision + se + 1e-6)

                # Compute AUC score (requires at least two distinct classes)
                if len(np.unique(target_valid)) < 2:
                    auc = 0.5
                else:
                    try:
                        auc = roc_auc_score(target_valid, pred_prob_valid)
                    except:
                        auc = 0.5                # 累加指标
                total_acc += acc
                total_se += se
                total_sp += sp
                total_f1 += f1
                total_auc += auc
                total_samples += 1


        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

        # 计算平均指标
    if total_samples > 0:
        avg_acc = total_acc / total_samples
        avg_se = total_se / total_samples
        avg_sp = total_sp / total_samples
        avg_f1 = total_f1 / total_samples
        avg_auc = total_auc / total_samples
    else:
        avg_acc = avg_se = avg_sp = avg_f1 = avg_auc = 0.0

        # 返回所有指标
    return confmat, dice.value.item(), {
            'acc': avg_acc,
            'se': avg_se,
            'sp': avg_sp,
            'f1': avg_f1,
            'auc': avg_auc
        }


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.amp.autocast(device_type='cuda',enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)