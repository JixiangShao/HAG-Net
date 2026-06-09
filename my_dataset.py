import os
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset


# my_dataset.py（补充HRF数据集类）
# 新增到my_dataset.py中
class HRFDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None, train_indices=None):
        super(HRFDataset, self).__init__()
        # HRF数据集无training/test子目录，直接使用根目录下的三个文件夹
        self.data_root = os.path.join(root, "HRF")
        assert os.path.exists(self.data_root), f"路径 '{self.data_root}' 不存在，请检查HRF数据集路径"

        self.transforms = transforms
        # 获取images文件夹下的所有图像文件（支持常见图像后缀）
        img_dir = os.path.join(self.data_root, "images")
        self.img_names = [i for i in os.listdir(img_dir)
                     if i.lower().endswith((".JPG", ".jpg", ".jpeg", ".tif"))]


        if train_indices is None:
        # 提取文件名中的数字前缀（如"01_dr.JPG" → "01"）
            indices = sorted(list(set([name.split("_")[0] for name in self.img_names])))
            split_idx = int(len(indices) * 0.8)  # 80%作为训练集
            self.train_indices = indices[:split_idx]
            self.test_indices = indices[split_idx:]
        else:
            self.train_indices = train_indices
            self.test_indices = [idx for idx in indices if idx not in train_indices]

        # 根据train参数筛选当前子集的图像
        if train:
            self.img_names = [name for name in self.img_names
                          if name.split("_")[0] in self.train_indices]
        else:
            self.img_names = [name for name in self.img_names
                          if name.split("_")[0] in self.test_indices]

        # 构建图像、标注、掩码的完整路径
        self.img_list = [os.path.join(img_dir, name) for name in self.img_names]

        manual_dir = os.path.join(self.data_root, "manual1")
        self.manual = [os.path.join(manual_dir, f"{name.split('.')[0]}.tif")
                   for name in self.img_names]  # 匹配manual1中的.tif标注

        mask_dir = os.path.join(self.data_root, "mask")
        self.roi_mask = [os.path.join(mask_dir, f"{name.split('.')[0]}_mask.tif")
                     for name in self.img_names]  # 匹配mask中的_mask.tif

    # 检查文件是否存在
        for path in self.manual + self.roi_mask:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在: {path}")

    def __getitem__(self, idx):
        # 读取原始图像（转为RGB）
        img = Image.open(self.img_list[idx]).convert('RGB')

        # 读取血管标注（转为灰度图）并归一化到0-1
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255.0

        # 读取ROI掩码（转为灰度图）并处理（反转掩码，使ROI区域为255）
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)  # 与DRIVE处理方式一致

        # 合并标注和ROI掩码（仅保留ROI区域内的标注）
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        mask = Image.fromarray(mask.astype(np.uint8))  # 转回PIL图像用于transforms处理

        # 应用数据增强
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        """用于DataLoader的批量处理函数，与DRIVE保持一致"""
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    """辅助函数：将不同尺寸的图像拼接成批量，与DRIVE保持一致"""
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs



class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # check files
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs



class CHASEDB1Dataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(CHASEDB1Dataset, self).__init__()
        # 根据train参数，确定是加载训练数据还是测试数据
        self.flag = "train" if train else "test"
        # 拼接数据集的根路径和对应的子路径（train或test）
        data_root = os.path.join(root, "CHASEDB1", self.flag)
        # 检查路径是否存在
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.transforms = transforms  # 图像变换操作
        # 获取图像文件名列表，仅保留扩展名为.jpg的文件
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
        manual_names = [i for i in os.listdir(os.path.join(data_root, "manual")) if i.endswith("1stHO.png")]
        mask_names = [i for i in os.listdir(os.path.join(data_root, "mask")) if i.endswith("_mask.gif")]

        # 生成图像路径列表
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # 生成mask1路径列表，文件名格式为"图像名_1stHO.png"
        self.manual = [os.path.join(data_root, "manual", i.split(".")[0] + "1stHO.png")
                           for i in img_names]
        self.manual = [os.path.join(data_root, "manual", i) for i in manual_names]

        # 生成mask2路径列表，文件名格式为"图像名_2ndHO.png"
        self.roi_mask = [os.path.join(data_root, "mask", i)
                           for i in mask_names]

        # 检查mask1和mask2文件是否存在
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exist.")

    def __getitem__(self, idx):
        # 打开第idx个图像并转换为RGB格式
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 打开对应的mask1和mask2并转换为灰度图（L模式）
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual)/255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        mask = Image.fromarray(mask)


        # 如果定义了transforms操作，则对图像和mask进行变换
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # 返回处理后的图像和对应的mask
        return img, mask

    def __len__(self):
        # 返回数据集的大小，即图像列表的长度
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        # 将一个batch中的图像和标签分别打包
        images, targets = list(zip(*batch))
        # 将图像打包为一个张量，缺失的部分用0填充
        batched_imgs = cat_list(images, fill_value=0)
        # 将标签打包为一个张量，缺失的部分用255填充
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):
    # 计算批次中每个维度的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # 生成包含所有图像的张量，形状为（批次数量, 最大尺寸）
    batch_shape = (len(images),) + max_size
    # 创建一个用fill_value填充的新张量
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # 将每个图像复制到对应的张量中，保持原有图像尺寸
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs  # 返回打包好的张量


class StareDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(StareDataset, self).__init__()

        self.flag = "train" if train else "test"
        # 拼接数据集的根路径和对应的子路径（train或test）
        data_root = os.path.join(root, "STARE", self.flag)
        # 检查路径是否存在
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.transforms = transforms  # 图像变换操作
        # 获取图像文件名列表，仅保留扩展名为.jpg的文件
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".ppm")]
        manual_names = [i for i in os.listdir(os.path.join(data_root, "manual2")) if i.endswith(".ah.ppm")]
        mask_names = [i for i in os.listdir(os.path.join(data_root, "mask")) if i.endswith("_mask.gif")]

        # 生成图像路径列表
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # 生成mask1路径列表，文件名格式为"图像名_1stHO.png"
        self.manual = [os.path.join(data_root, "manual2", i.split(".")[0] + ".ah.ppm")
                       for i in img_names]
        self.manual = [os.path.join(data_root, "manual2", i) for i in manual_names]

        # 生成mask2路径列表，文件名格式为"图像名_2ndHO.png"
        self.roi_mask = [os.path.join(data_root, "mask", i)
                         for i in mask_names]

        # 检查mask1和mask2文件是否存在
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exist.")

    def __getitem__(self, idx):
        # 打开第idx个图像并转换为RGB格式
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 打开对应的mask1和mask2并转换为灰度图（L模式）
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        mask = Image.fromarray(mask)

        # 如果定义了transforms操作，则对图像和mask进行变换
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # 返回处理后的图像和对应的mask
        return img, mask

    def __len__(self):
        # 返回数据集的大小，即图像列表的长度
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        # 将一个batch中的图像和标签分别打包
        images, targets = list(zip(*batch))
        # 将图像打包为一个张量，缺失的部分用0填充
        batched_imgs = cat_list(images, fill_value=0)
        # 将标签打包为一个张量，缺失的部分用255填充
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算批次中每个维度的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # 生成包含所有图像的张量，形状为（批次数量, 最大尺寸）
    batch_shape = (len(images),) + max_size
    # 创建一个用fill_value填充的新张量
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # 将每个图像复制到对应的张量中，保持原有图像尺寸
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs  # 返回打包好的张量