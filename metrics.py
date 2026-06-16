import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_metrics(pred, gt, roi=None):
    """
    Calculate ACC, SE, SP, Dice, mIoU, F1, AUC metrics for a single sample, supporting ROI region filtering.

    Parameters:
    --------
    pred (np.ndarray): Predicted binary mask or probability map, shape (H, W), values 0/1 (binary) or [0,1] (probability, for AUC).
    gt (np.ndarray):   Ground truth label mask, shape (H, W), values 0 (background) or 1 (foreground).
    roi (np.ndarray, optional): ROI mask, shape (H, W), values True (valid region) or False (invalid region).
                           If None, the entire image region is calculated by default.

    Returns:
    -------
    acc (float): Accuracy, range [0, 1].
    se (float):  Sensitivity (foreground recall), range [0, 1].
    sp (float):  Specificity (background precision), range [0, 1].
    dice (float): Dice coefficient (foreground class), range [0, 1].
    mIoU (float): Mean Intersection over Union (average of background + foreground classes), range [0, 1].
    f1 (float):   F1-score (harmonic mean of precision and recall), range [0, 1].
    auc (float):  Area Under ROC Curve, range [0, 1] (more meaningful when input is a probability map).
    """
    # Save original prediction values for AUC calculation (retain float type if it's a probability map)
    pred_auc = pred.copy()

    # Convert masks to boolean type for other metric calculations (binarization)
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)

    # Apply ROI filtering (only calculate valid regions, exclude irrelevant background)
    if roi is not None:
        pred_bin = pred_bin[roi]
        gt_bin = gt_bin[roi]
        pred_auc = pred_auc[roi]
        gt_auc = gt[roi]

    # 1. Calculate four core elements of the confusion matrix (binary classification scenario)
    tp = np.sum(pred_bin & gt_bin)  # True Positive: correctly predicted foreground (1→1)
    tn = np.sum(~pred_bin & ~gt_bin)  # True Negative: correctly predicted background (0→0)
    fp = np.sum(pred_bin & ~gt_bin)  # False Positive: background misclassified as foreground (0→1)
    fn = np.sum(~pred_bin & gt_bin)  # False Negative: foreground misclassified as background (1→0)

    # 2. Calculate original metrics (ACC, SE, SP, Dice)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6) if (tp + tn + fp + fn) != 0 else 0.0
    se = tp / (tp + fn + 1e-6) if (tp + fn) != 0 else 0.0  # Recall (same as SE)
    sp = tn / (tn + fp + 1e-6) if (tn + fp) != 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6) if (2 * tp + fp + fn) != 0 else 0.0

    # 3. Calculate mIoU
    foreground_iou = tp / (tp + fp + fn + 1e-6) if (tp + fp + fn) != 0 else 0.0
    background_iou = tn / (tn + fp + fn + 1e-6) if (tn + fp + fn) != 0 else 0.0
    mIoU = (background_iou + foreground_iou) / 2.0

    # 4. F1-score (harmonic mean of precision and recall)
    precision = tp / (tp + fp + 1e-6) if (tp + fp) != 0 else 0.0  # Precision
    f1 = 2 * (precision * se) / (precision + se + 1e-6) if (precision + se) != 0 else 0.0

    # 5. Calculate AUC (use original prediction values, supports probability input)
    # Handle edge cases: AUC is meaningless when all samples belong to the same class, return 0.5
    if len(np.unique(gt_auc)) < 2:
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(gt_auc, pred_auc)
        except:
            auc = 0.5  # Return default value if calculation fails

    return acc, se, sp, dice, mIoU, f1, auc