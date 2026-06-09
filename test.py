import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from src import UNet,ResNetUNet,VGG16UNet,MobileV3Unet,MMUNet,SegNet,R2U_Net,NestedUNet,get_fcn_model,create_ce_net,UNet_HLAEM,UNet_HLAEM_GNN
import datetime
import torch.serialization
import argparse
from sklearn.metrics import roc_auc_score  # 新增：用于AUC计算

# -------------------------- 配置参数（修改日志保存路径） --------------------------
data_root = "./DRIVE"
weights_path = "./save_weights/UNet_HLAEM_GNN__best_model.pth"
num_classes = 1  # 二分类（背景+前景）
mean = (0.709, 0.381, 0.224)
std = (0.127, 0.079, 0.043)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_pred = True
save_dir = "results_DRIVE/test_pred_UNet_HLAEM_GNN_2"  # 预测图像保存目录（如需修改可同步调整）

log_file_path = "results_DRIVE/test_pred_UNet_HLAEM_GNN_2/test_results_UNet_HLAEM_GNN_2.txt"


# -------------------------- 辅助函数（更新指标计算，添加F1和AUC） --------------------------
def calculate_metrics(pred_roi, gt_roi, pred_prob_roi):
    """计算ACC、SE、SP、Dice、mIoU、F1、AUC指标（ROI区域内）"""
    # 混淆矩阵核心元素
    tp = np.sum((pred_roi == 1) & (gt_roi == 1))  # 真阳性
    tn = np.sum((pred_roi == 0) & (gt_roi == 0))  # 真阴性
    fp = np.sum((pred_roi == 1) & (gt_roi == 0))  # 假阳性
    fn = np.sum((pred_roi == 0) & (gt_roi == 1))  # 假阴性

    # 原有指标
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6) if (tp + tn + fp + fn) != 0 else 0.0
    se = tp / (tp + fn + 1e-6) if (tp + fn) != 0 else 0.0  # 敏感性（召回率）
    sp = tn / (tn + fp + 1e-6) if (tn + fp) != 0 else 0.0  # 特异性
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6) if (2 * tp + fp + fn) != 0 else 0.0

    # mIoU
    foreground_iou = tp / (tp + fp + fn + 1e-6) if (tp + fp + fn) != 0 else 0.0
    background_iou = tn / (tn + fp + fn + 1e-6) if (tn + fp + fn) != 0 else 0.0
    mIoU = (background_iou + foreground_iou) / 2.0

    # 新增：F1分数（精确率与召回率的调和平均）
    precision = tp / (tp + fp + 1e-6) if (tp + fp) != 0 else 0.0  # 精确率
    f1 = 2 * (precision * se) / (precision + se + 1e-6) if (precision + se) != 0 else 0.0

    # 新增：AUC（ROC曲线下面积，基于预测概率）
    # 处理极端情况：所有标签为同一类别时AUC无意义，返回0.5
    if len(np.unique(gt_roi)) < 2:
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(gt_roi, pred_prob_roi)  # 使用前景概率计算
        except:
            auc = 0.5  # 计算失败时返回默认值

    return acc, se, sp, dice, mIoU, f1, auc


def write_log(content, log_path):
    print(content)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")


# -------------------------- 主流程 --------------------------
if __name__ == "__main__":
    # 1. 初始化目录（确保日志保存目录存在）
    if save_pred and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # 确保 results_DRIVE 目录存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 2. 日志头部
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_header = f"==================== 测试开始 ====================\n" \
                 f"测试时间：{start_time}\n" \
                 f"使用设备：{device}\n" \
                 f"模型权重路径：{weights_path}\n" \
                 f"测试集根路径：{data_root}\n" \
                 f"预测结果保存目录：{save_dir}\n" \
                 f"当前日志文件名：{os.path.basename(log_file_path)}\n" \
                 "=======================================================\n"
    write_log(log_header, log_file_path)
    torch.serialization.add_safe_globals([argparse.Namespace])
    # 3. 加载模型
    # model = UNet(in_channels=3, num_classes=num_classes + 1)  # 输出通道：背景+前景=2
    # model = ResNetUNet(num_classes=2)
    # model = VGG16UNet(num_classes=2, pretrain_backbone=False)
    # model = MobileV3Unet(num_classes=2)
    # model = MMUNet(in_channels=3, num_classes=num_classes+1)
    # model = SegNet(in_channels=3,num_classes=2)
    # model = R2U_Net(in_channels=3,num_classes=2,t=2)
    # model = NestedUNet(in_channels=3,num_classes=2,deepsupervision=False)
    # model = get_fcn_model(model_type='fcn8s', num_classes=2, backbone='vgg16', pretrained=True)
    # model = create_ce_net(model_type="base", num_classes=2, pretrained=True)
    # model = UNet_HLAEM(in_channels=3,num_classes=2)
    model = UNet_HLAEM_GNN(in_channels=3,num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True)['model'])
    model.to(device).eval()
    write_log(f"模型加载成功！ 输入通道：3，输出类别数：{num_classes + 1}", log_file_path)

    # 4. 预处理和数据集路径
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_img_dir = os.path.join(data_root, "test", "images")
    test_roi_dir = os.path.join(data_root, "test", "mask")
    test_gt_dir = os.path.join(data_root, "test", "1st_manual")

    # 5. 检查测试样本
    img_names = [f for f in os.listdir(test_img_dir) if f.endswith(".tif")]
    if not img_names:
        error_msg = f"未找到测试图像，请检查路径：{test_img_dir}"
        write_log(error_msg, log_file_path)
        raise FileNotFoundError(error_msg)
    write_log(f"\n共找到 {len(img_names)} 个测试样本，开始推理...\n", log_file_path)

    # 6. 初始化指标累加变量（新增F1和AUC）
    total_acc, total_se, total_sp, total_dice, total_mIoU = 0.0, 0.0, 0.0, 0.0, 0.0
    total_f1, total_auc = 0.0, 0.0  # 新增：F1和AUC累加

    # ==================================================
    # 样本循环
    # ==================================================
    for img_idx, img_name in enumerate(img_names, 1):
        # 6.1 读取单样本数据
        img_path = os.path.join(test_img_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        roi_name = img_name.replace("_test.tif", "_test_mask.gif")
        roi_path = os.path.join(test_roi_dir, roi_name)
        roi_mask = Image.open(roi_path).convert('L')
        roi = np.array(roi_mask) == 255  # ROI区域（有效区域为True）

        gt_name = img_name.replace("_test.tif", "_manual1.gif")
        gt_path = os.path.join(test_gt_dir, gt_name)
        gt_mask = Image.open(gt_path).convert('L')
        gt = (np.array(gt_mask) == 255).astype(np.uint8)  # 真实标签（1为前景，0为背景）

        # 6.2 单样本推理（新增：获取前景概率用于AUC）
        img_tensor = data_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            logits = outputs['out']  # 原始输出（未经过softmax）
            pred_prob = torch.softmax(logits, dim=1)[:, 1, ...].squeeze(0).cpu().numpy()  # 前景概率（AUC用）
            pred = logits.argmax(1).squeeze(0).cpu().numpy()  # 二值预测（0/1，其他指标用）

        # 6.3 单样本指标计算与累加（新增F1和AUC）
        acc, se, sp, dice, mIoU, f1, auc = calculate_metrics(
            pred_roi=pred[roi],  # 二值预测的ROI区域
            gt_roi=gt[roi],  # 真实标签的ROI区域
            pred_prob_roi=pred_prob[roi]  # 前景概率的ROI区域（AUC用）
        )
        # 累加所有指标
        total_acc += acc
        total_se += se
        total_sp += sp
        total_dice += dice
        total_mIoU += mIoU
        total_f1 += f1  # 新增
        total_auc += auc  # 新增

        # 6.4 单样本日志（包含新指标）
        sample_log = f"样本 {img_idx}/{len(img_names)}：{img_name}\n" \
                     f"  ACC:  {acc:.4f} | SE:   {se:.4f} | SP:   {sp:.4f}\n" \
                     f"  Dice: {dice:.4f} | mIoU: {mIoU:.4f} | F1:   {f1:.4f}\n" \
                     f"  AUC:  {auc:.4f}"
        write_log(sample_log, log_file_path)

        # 6.5 单样本预测结果保存
        if save_pred:
            pred_mask = np.zeros_like(roi, dtype=np.uint8)
            pred_mask[roi] = pred[roi] * 255  # ROI内的预测结果（0/255）
            save_path = os.path.join(save_dir, img_name.replace(".tif", "_pred.png"))
            Image.fromarray(pred_mask).save(save_path)
            write_log(f"  预测结果已保存至：{save_path}\n", log_file_path)

    # ==================================================
    # 全局结果计算与打印（包含新指标）
    # ==================================================
    write_log("\n" + "=" * 50, log_file_path)
    if len(img_names) > 0:
        # 计算全局平均
        avg_acc = total_acc / len(img_names)
        avg_se = total_se / len(img_names)
        avg_sp = total_sp / len(img_names)
        avg_dice = total_dice / len(img_names)
        avg_mIoU = total_mIoU / len(img_names)
        avg_f1 = total_f1 / len(img_names)  # 新增
        avg_auc = total_auc / len(img_names)  # 新增

        # 全局结果日志
        global_log = "测试集全局结果\n" \
                     "==============================\n" \
                     f"测试样本总数：{len(img_names)} 个\n" \
                     f"平均ACC:  {avg_acc:.4f} （所有像素预测正确率）\n" \
                     f"平均SE:   {avg_se:.4f} （敏感性/前景召回率）\n" \
                     f"平均SP:   {avg_sp:.4f} （特异性/背景精确率）\n" \
                     f"平均Dice: {avg_dice:.4f} （前景类重叠度）\n" \
                     f"平均mIoU: {avg_mIoU:.4f} （背景+前景平均交并比）\n" \
                     f"平均F1:   {avg_f1:.4f} （精确率与召回率的调和平均）\n" \
                     f"平均AUC:  {avg_auc:.4f} （ROC曲线下面积，衡量排序能力）\n" \
                     "=================================="
        write_log(global_log, log_file_path)
    else:
        write_log("未处理任何测试样本！", log_file_path)

    # 7. 测试结束日志
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_log = f"\n==================== 测试结束 ====================\n" \
              f"结束时间：{end_time}\n" \
              f"本次测试日志已保存至：{log_file_path}\n" \
              f"日志文件名：{os.path.basename(log_file_path)}\n" \
              "======================================================="
    write_log(end_log, log_file_path)