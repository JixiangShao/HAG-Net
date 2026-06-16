import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from src import UNet,ResNetUNet,SegNet,R2U_Net,NestedUNet,get_fcn_model,create_ce_net,HAG_Net
import datetime
import torch.serialization
import argparse
from sklearn.metrics import roc_auc_score

# -------------------------- Configuration Parameters (Modify log save path) --------------------------
data_root = "./DRIVE"
weights_path = "./save_weights_DRIVE/HAG_Net_model.pth"
num_classes = 1  # Binary segmentation (background + foreground vessel)
mean = (0.709, 0.381, 0.224)
std = (0.127, 0.079, 0.043)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_pred = True
save_dir = "results_DRIVE/test_pred_HAG_Net"  # Directory to store predicted segmentation images (modify here if needed)

log_file_path = "results_DRIVE/test_pred_HAG_Net/test_results_HAG_Net.txt"


# -------------------------- Helper Functions (Updated metric calculation logic) --------------------------
def calculate_metrics(pred_roi, gt_roi, pred_prob_roi):
    """Calculate ACC, SE, SP, Dice, mIoU, F1, AUC metrics within valid ROI region"""
    # Core elements of confusion matrix
    tp = np.sum((pred_roi == 1) & (gt_roi == 1))  # True Positive
    tn = np.sum((pred_roi == 0) & (gt_roi == 0))  # True Negative
    fp = np.sum((pred_roi == 1) & (gt_roi == 0))  # False Positive
    fn = np.sum((pred_roi == 0) & (gt_roi == 1))  # False Negative


    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6) if (tp + tn + fp + fn) != 0 else 0.0
    se = tp / (tp + fn + 1e-6) if (tp + fn) != 0 else 0.0  # Sensitivity (Recall for foreground)
    sp = tn / (tn + fp + 1e-6) if (tn + fp) != 0 else 0.0  # Specificity
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6) if (2 * tp + fp + fn) != 0 else 0.0

    # Mean Intersection over Union
    foreground_iou = tp / (tp + fp + fn + 1e-6) if (tp + fp + fn) != 0 else 0.0
    background_iou = tn / (tn + fp + fn + 1e-6) if (tn + fp + fn) != 0 else 0.0
    mIoU = (background_iou + foreground_iou) / 2.0

    # F1 Score (harmonic mean of precision and recall)
    precision = tp / (tp + fp + 1e-6) if (tp + fp) != 0 else 0.0  # Precision
    f1 = 2 * (precision * se) / (precision + se + 1e-6) if (precision + se) != 0 else 0.0

    # AUC (Area Under ROC Curve, calculated from predicted probability map)
    # Handle extreme case: return 0.5 when all labels belong to single class (AUC meaningless)
    if len(np.unique(gt_roi)) < 2:
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(gt_roi, pred_prob_roi)  # Use foreground probability values for computation
        except:
            auc = 0.5  # Default fallback value if calculation fails

    return acc, se, sp, dice, mIoU, f1, auc


def write_log(content, log_path):
    print(content)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")


# -------------------------- Main Test Pipeline --------------------------
if __name__ == "__main__":
    # 1. Initialize storage directories (ensure log directory exists)
    if save_pred and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Create parent directory for log file if missing
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 2. Write log header information
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_header = f"==================== Test Started ====================\n" \
                 f"Test Timestamp: {start_time}\n" \
                 f"Computation Device: {device}\n" \
                 f"Model Weight Path: {weights_path}\n" \
                 f"Test Dataset Root: {data_root}\n" \
                 f"Prediction Output Directory: {save_dir}\n" \
                 f"Current Log Filename: {os.path.basename(log_file_path)}\n" \
                 "=======================================================\n"
    write_log(log_header, log_file_path)
    torch.serialization.add_safe_globals([argparse.Namespace])
    # 3. Load segmentation model
    # model = UNet(in_channels=3, num_classes=num_classes + 1)  # Output channels: background + foreground = 2
    # model = ResNetUNet(num_classes=2)
    # model = SegNet(in_channels=3,num_classes=2)
    # model = R2U_Net(in_channels=3,num_classes=2,t=2)
    # model = NestedUNet(in_channels=3,num_classes=2,deepsupervision=False)
    # model = get_fcn_model(model_type='fcn8s', num_classes=2, backbone='vgg16', pretrained=True)
    # model = create_ce_net(model_type="base", num_classes=2, pretrained=True)
    model = HAG_Net(in_channels=3,num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True)['model'])
    model.to(device).eval()
    write_log(f"Model loaded successfully! Input channels: 3, Output classes: {num_classes + 1}", log_file_path)

    # 4. Image preprocessing pipeline and test dataset paths
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_img_dir = os.path.join(data_root, "test", "images")
    test_roi_dir = os.path.join(data_root, "test", "mask")
    test_gt_dir = os.path.join(data_root, "test", "1st_manual")

    # 5. Scan and validate test image samples
    img_names = [f for f in os.listdir(test_img_dir) if f.endswith(".tif")]
    if not img_names:
        error_msg = f"No test images found, please check directory path: {test_img_dir}"
        write_log(error_msg, log_file_path)
        raise FileNotFoundError(error_msg)
    write_log(f"\nTotal {len(img_names)} test samples detected, starting inference...\n", log_file_path)

    # 6. Initialize accumulators for all evaluation metrics
    total_acc, total_se, total_sp, total_dice, total_mIoU = 0.0, 0.0, 0.0, 0.0, 0.0
    total_f1, total_auc = 0.0, 0.0  # New accumulators for F1 and AUC metrics

    # ==================================================
    # Iterate over all test samples
    # ==================================================
    for img_idx, img_name in enumerate(img_names, 1):
        # 6.1 Load single sample data
        img_path = os.path.join(test_img_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        roi_name = img_name.replace("_test.tif", "_test_mask.gif")
        roi_path = os.path.join(test_roi_dir, roi_name)
        roi_mask = Image.open(roi_path).convert('L')
        roi = np.array(roi_mask) == 255  # Valid ROI region (True = area to evaluate)

        gt_name = img_name.replace("_test.tif", "_manual1.gif")
        gt_path = os.path.join(test_gt_dir, gt_name)
        gt_mask = Image.open(gt_path).convert('L')
        gt = (np.array(gt_mask) == 255).astype(np.uint8)  # Ground truth label (1=vessel foreground, 0=background)

        # 6.2 Single image inference (extract foreground probability map for AUC calculation)
        img_tensor = data_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            logits = outputs['out']  # Raw model output before softmax
            pred_prob = torch.softmax(logits, dim=1)[:, 1, ...].squeeze(0).cpu().numpy()  # Foreground probability for AUC
            pred = logits.argmax(1).squeeze(0).cpu().numpy()  # Binary segmentation prediction (0/1) for other metrics

        # 6.3 Calculate and accumulate metrics for current sample (including F1 and AUC)
        acc, se, sp, dice, mIoU, f1, auc = calculate_metrics(
            pred_roi=pred[roi],  # Binary prediction cropped to valid ROI
            gt_roi=gt[roi],  # Ground truth cropped to valid ROI
            pred_prob_roi=pred_prob[roi]  # Foreground probability cropped to valid ROI for AUC
        )
        # Accumulate all metric values
        total_acc += acc
        total_se += se
        total_sp += sp
        total_dice += dice
        total_mIoU += mIoU
        total_f1 += f1  # New metric accumulation
        total_auc += auc  # New metric accumulation

        # 6.4 Write log for single test sample with full metrics
        sample_log = f"Sample {img_idx}/{len(img_names)}: {img_name}\n" \
                     f"  ACC:  {acc:.4f} | SE:   {se:.4f} | SP:   {sp:.4f}\n" \
                     f"  Dice: {dice:.4f} | mIoU: {mIoU:.4f} | F1:   {f1:.4f}\n" \
                     f"  AUC:  {auc:.4f}"
        write_log(sample_log, log_file_path)

        # 6.5 Save predicted segmentation mask for current sample
        if save_pred:
            pred_mask = np.zeros_like(roi, dtype=np.uint8)
            pred_mask[roi] = pred[roi] * 255  # Convert prediction to 0/255 within ROI area
            save_path = os.path.join(save_dir, img_name.replace(".tif", "_pred.png"))
            Image.fromarray(pred_mask).save(save_path)
            write_log(f"  Prediction mask saved to: {save_path}\n", log_file_path)

    # ==================================================
    # Compute and print overall dataset average metrics
    # ==================================================
    write_log("\n" + "=" * 50, log_file_path)
    if len(img_names) > 0:
        # Calculate average metrics across all test samples
        avg_acc = total_acc / len(img_names)
        avg_se = total_se / len(img_names)
        avg_sp = total_sp / len(img_names)
        avg_dice = total_dice / len(img_names)
        avg_mIoU = total_mIoU / len(img_names)
        avg_f1 = total_f1 / len(img_names)  # New average F1
        avg_auc = total_auc / len(img_names)  # New average AUC

        # Log global average results
        global_log = "Overall Test Set Performance\n" \
                     "==============================\n" \
                     f"Total Test Samples: {len(img_names)}\n" \
                     f"Average ACC:  {avg_acc:.4f}  (Global pixel-wise prediction accuracy)\n" \
                     f"Average SE:   {avg_se:.4f}  (Sensitivity / Foreground Recall)\n" \
                     f"Average SP:   {avg_sp:.4f}  (Specificity / Background Precision)\n" \
                     f"Average Dice: {avg_dice:.4f}  (Foreground overlap coefficient)\n" \
                     f"Average mIoU: {avg_mIoU:.4f}  (Mean IoU of background and foreground)\n" \
                     f"Average F1:   {avg_f1:.4f}  (Harmonic mean of precision and recall)\n" \
                     f"Average AUC:  {avg_auc:.4f}  (Area under ROC curve, ranking discriminability)\n" \
                     "=================================="
        write_log(global_log, log_file_path)
    else:
        write_log("No test samples were processed!", log_file_path)

    # 7. Write test completion footer log
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_log = f"\n==================== Test Finished ====================\n" \
              f"Completion Timestamp: {end_time}\n" \
              f"Full test log saved at: {log_file_path}\n" \
              f"Log File Name: {os.path.basename(log_file_path)}\n" \
              "======================================================="
    write_log(end_log, log_file_path)