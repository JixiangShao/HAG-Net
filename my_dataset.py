import os
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        # Set dataset mode: training or test
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE", self.flag)
        # Verify the dataset path exists
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        # Get all .tif image filenames
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        # Generate full paths for images
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # Generate full paths for manual segmentation labels
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # Check if manual label files exist
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # Generate full paths for ROI masks
        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # Check if ROI mask files exist
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        # Open image and convert to RGB format
        img = Image.open(self.img_list[idx]).convert('RGB')
        # Open manual segmentation label and convert to grayscale
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        # Open ROI mask and convert to grayscale
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        # Combine manual label and ROI mask to generate final mask
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # Convert back to PIL Image because transforms process PIL data
        mask = Image.fromarray(mask)

        # Apply transformations if provided
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        # Return the total number of samples
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        # Unpack batch into images and targets
        images, targets = list(zip(*batch))
        # Batch images with padding value 0
        batched_imgs = cat_list(images, fill_value=0)
        # Batch targets with padding value 255
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # Get maximum size across all images in the batch
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # Create batch tensor shape
    batch_shape = (len(images),) + max_size
    # Initialize batch tensor with fill value
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # Copy each image into the padded batch tensor
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class CHASEDB1Dataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(CHASEDB1Dataset, self).__init__()
        # Determine whether to load training or test data based on the train parameter
        self.flag = "train" if train else "test"
        # Concatenate the root path and corresponding sub-path (train or test)
        data_root = os.path.join(root, "CHASEDB1", self.flag)
        # Check if the path exists
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.transforms = transforms  # Image transformation operations
        # Get image filename list, only keep files with .jpg extension
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
        manual_names = [i for i in os.listdir(os.path.join(data_root, "manual")) if i.endswith("1stHO.png")]
        mask_names = [i for i in os.listdir(os.path.join(data_root, "mask")) if i.endswith("_mask.gif")]

        # Generate image path list
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # Generate manual label path list
        self.manual = [os.path.join(data_root, "manual", i.split(".")[0] + "1stHO.png")
                           for i in img_names]
        self.manual = [os.path.join(data_root, "manual", i) for i in manual_names]

        # Generate ROI mask path list
        self.roi_mask = [os.path.join(data_root, "mask", i)
                           for i in mask_names]

        # Check if manual label files exist
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exist.")

    def __getitem__(self, idx):
        # Open the idx-th image and convert to RGB format
        img = Image.open(self.img_list[idx]).convert('RGB')
        # Open corresponding manual label and ROI mask, convert to grayscale (L mode)
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual)/255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        # Combine label and mask to generate final segmentation mask
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        mask = Image.fromarray(mask)

        # Apply transformations if defined
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # Return processed image and corresponding mask
        return img, mask

    def __len__(self):
        # Return dataset size, i.e., length of image list
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        # Separate images and targets in a batch
        images, targets = list(zip(*batch))
        # Batch images with 0 padding for missing parts
        batched_imgs = cat_list(images, fill_value=0)
        # Batch targets with 255 padding for missing parts
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):
    # Calculate maximum size for each dimension in the batch
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # Create tensor shape containing all images
    batch_shape = (len(images),) + max_size
    # Create new tensor filled with fill_value
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # Copy each image to corresponding tensor while preserving original size
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs  # Return batched tensor


class StareDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(StareDataset, self).__init__()

        self.flag = "train" if train else "test"
        # Concatenate the root path and corresponding sub-path (train or test)
        data_root = os.path.join(root, "STARE", self.flag)
        # Check if the path exists
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.transforms = transforms  # Image transformation operations
        # Get image filename list, only keep files with .ppm extension
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".ppm")]
        manual_names = [i for i in os.listdir(os.path.join(data_root, "manual2")) if i.endswith(".ah.ppm")]
        mask_names = [i for i in os.listdir(os.path.join(data_root, "mask")) if i.endswith("_mask.gif")]

        # Generate image path list
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # Generate manual label path list
        self.manual = [os.path.join(data_root, "manual2", i.split(".")[0] + ".ah.ppm")
                       for i in img_names]
        self.manual = [os.path.join(data_root, "manual2", i) for i in manual_names]

        # Generate ROI mask path list
        self.roi_mask = [os.path.join(data_root, "mask", i)
                         for i in mask_names]

        # Check if manual label files exist
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exist.")

    def __getitem__(self, idx):
        # Open the idx-th image and convert to RGB format
        img = Image.open(self.img_list[idx]).convert('RGB')
        # Open corresponding manual label and ROI mask, convert to grayscale (L mode)
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        # Combine label and mask to generate final segmentation mask
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        mask = Image.fromarray(mask)

        # Apply transformations if defined
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # Return processed image and corresponding mask
        return img, mask

    def __len__(self):
        # Return dataset size, i.e., length of image list
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        # Separate images and targets in a batch
        images, targets = list(zip(*batch))
        # Batch images with 0 padding for missing parts
        batched_imgs = cat_list(images, fill_value=0)
        # Batch targets with 255 padding for missing parts
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # Calculate maximum size for each dimension in the batch
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # Create tensor shape containing all images
    batch_shape = (len(images),) + max_size
    # Create new tensor filled with fill_value
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # Copy each image to corresponding tensor while preserving original size
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs  # Return batched tensor