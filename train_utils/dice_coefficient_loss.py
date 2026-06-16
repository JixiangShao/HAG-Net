import torch
import torch.nn as nn
import torch.nn.functional as F


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """Generate one-hot target tensor for Dice loss calculation"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # Transform shape from [N, H, W] to [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Calculate average Dice coefficient for all images in a single batch
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # Create ROI mask to exclude pixels labeled as ignore_index
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Compute average Dice coefficient across all segmentation classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss value to be minimized, range between 0 and 1
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)


# Boundary loss function
def boundary_loss(pred, target, ignore_index=-100):
    """Optimized boundary-aware loss function"""
    # Unify data type of input tensors
    if target.dtype != torch.float32:
        target = target.float()

    batch_size = pred.shape[0]

    # Replace Sobel operator with Prewitt operator for more stable edge extraction
    kernel_x = torch.tensor([[[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]]],
                            dtype=torch.float32, device=pred.device)
    kernel_y = torch.tensor([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]],
                            dtype=torch.float32, device=pred.device)

    total_boundary_loss = 0.0

    for i in range(batch_size):
        # Extract edge map from ground truth label
        target_single = target[i:i + 1].unsqueeze(1)  # Reshape to [1, 1, H, W]
        target_edges_x = F.conv2d(target_single, kernel_x, padding=1)
        target_edges_y = F.conv2d(target_single, kernel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2 + 1e-8)
        target_edges = (target_edges > 0.1).float()

        # Extract foreground probability map from model prediction
        pred_prob = torch.softmax(pred[i:i + 1], dim=1)[:, 1:2]  # Reshape to [1, 1, H, W]

        # Calculate Dice loss on boundary regions only
        intersection = (pred_prob * target_edges).sum()
        union = pred_prob.sum() + target_edges.sum()
        boundary_dice = (2. * intersection + 1e-6) / (union + 1e-6)

        total_boundary_loss += (1 - boundary_dice)

    return total_boundary_loss / batch_size


def total_variation_loss(pred, weight=1e-4):
    """Total Variation loss to smooth prediction maps and reduce segmentation discontinuities"""
    pred_prob = torch.softmax(pred, dim=1)[:, 1:2]  # Extract foreground probability channel

    # Compute pixel differences along horizontal and vertical axes
    h_tv = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :]).mean()
    w_tv = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1]).mean()

    return weight * (h_tv + w_tv)