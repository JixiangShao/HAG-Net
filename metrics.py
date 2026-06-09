import numpy as np
from sklearn.metrics import roc_auc_score  # 用于计算AUC


def calculate_metrics(pred, gt, roi=None):
    """
    计算单样本的 ACC、SE、SP、Dice、mIoU、F1、AUC 指标，支持 ROI 区域过滤（二分类场景专用）。

    参数说明：
    --------
    pred (np.ndarray): 预测的二值掩码或概率图，形状 (H, W)，值为 0/1（二值）或 [0,1] 区间（概率，用于AUC）。
    gt (np.ndarray):   真实标签掩码，形状 (H, W)，值为 0（背景）或 1（前景）。
    roi (np.ndarray, 可选): ROI 掩码，形状 (H, W)，值为 True（有效区域）或 False（无效区域）。
                           若为 None，默认计算整个图像区域。

    返回值：
    -------
    acc (float): 准确率，范围 [0, 1]。
    se (float):  敏感性（前景召回率），范围 [0, 1]。
    sp (float):  特异性（背景精确率），范围 [0, 1]。
    dice (float): Dice 系数（前景类），范围 [0, 1]。
    mIoU (float): 平均交并比（背景+前景类平均），范围 [0, 1]。
    f1 (float):   F1分数（精确率与召回率的调和平均），范围 [0, 1]。
    auc (float):  ROC曲线下面积，范围 [0, 1]（需输入概率图时更有意义）。
    """
    # 保存原始预测值用于AUC计算（若为概率图则保留浮点型）
    pred_auc = pred.copy()

    # 将掩码转换为布尔类型用于其他指标计算（二值化处理）
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)

    # 应用 ROI 过滤（仅计算有效区域，排除无关背景）
    if roi is not None:
        pred_bin = pred_bin[roi]
        gt_bin = gt_bin[roi]
        pred_auc = pred_auc[roi]
        gt_auc = gt[roi]  # 真实标签保持0/1整数型用于AUC

    # 1. 计算混淆矩阵四大核心元素（二分类场景）
    tp = np.sum(pred_bin & gt_bin)  # 真阳性：前景预测正确（1→1）
    tn = np.sum(~pred_bin & ~gt_bin)  # 真阴性：背景预测正确（0→0）
    fp = np.sum(pred_bin & ~gt_bin)  # 假阳性：背景误判为前景（0→1）
    fn = np.sum(~pred_bin & gt_bin)  # 假阴性：前景误判为背景（1→0）

    # 2. 原有指标计算（ACC、SE、SP、Dice）
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6) if (tp + tn + fp + fn) != 0 else 0.0
    se = tp / (tp + fn + 1e-6) if (tp + fn) != 0 else 0.0  # 召回率（与SE一致）
    sp = tn / (tn + fp + 1e-6) if (tn + fp) != 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6) if (2 * tp + fp + fn) != 0 else 0.0

    # 3. mIoU 计算
    foreground_iou = tp / (tp + fp + fn + 1e-6) if (tp + fp + fn) != 0 else 0.0
    background_iou = tn / (tn + fp + fn + 1e-6) if (tn + fp + fn) != 0 else 0.0
    mIoU = (background_iou + foreground_iou) / 2.0

    # 4. 新增 F1 分数（精确率与召回率的调和平均）
    precision = tp / (tp + fp + 1e-6) if (tp + fp) != 0 else 0.0  # 精确率
    f1 = 2 * (precision * se) / (precision + se + 1e-6) if (precision + se) != 0 else 0.0

    # 5. 新增 AUC 计算（使用原始预测值，支持概率输入）
    # 处理极端情况：所有样本为同一类别时AUC无意义，返回0.5
    if len(np.unique(gt_auc)) < 2:
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(gt_auc, pred_auc)
        except:
            auc = 0.5  # 计算失败时返回默认值

    return acc, se, sp, dice, mIoU, f1, auc