import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import cv2
from typing import List, Optional, Tuple, Union


class BaseImageDataset(Dataset):
    """Base dataset class with common functionality"""
    
    def __init__(
        self,
        listroot: Optional[str] = None,
        labelroot: Optional[str] = None,
        shuffle: bool = True,
        transform: Optional[transforms.Compose] = None,
        depth_transform: Optional[transforms.Compose] = None
    ):
        self.listroot = listroot
        self.labelroot = labelroot
        self.transform = transform
        self.depth_transform = depth_transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ])
        
        with open(listroot, 'r') as file:
            self.lines = file.readlines()
            
        if shuffle:
            random.shuffle(self.lines)
            
        self.nSamples = len(self.lines)

    def __len__(self) -> int:
        return self.nSamples

    def _load_image(self, path: str) -> np.ndarray:
        """Load and convert BGR image to RGB"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _create_depth_label(self, label: int, size: Tuple[int, int] = (48, 48)) -> np.ndarray:
        """Create depth label based on class"""
        depth_label = np.ones(size) if label == 1 else np.zeros(size)
        return depth_label.astype(np.float32)

    def __getitem__(self, index: int):
        raise NotImplementedError


class MyDataSet(BaseImageDataset):
    """Basic RGB image dataset"""
    
    def __init__(self, images_path: List[str], images_class: List[int], transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError(f"Image {self.images_path[item]} isn't RGB mode")
            
        label = self.images_class[item]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default collate function"""
        images, labels = zip(*batch)
        return torch.stack(images, dim=0), torch.as_tensor(labels)


class MultiModalDataset(BaseImageDataset):
    """Dataset for RGB + HSV + YCbCr modalities"""
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        assert index <= len(self), 'index range error'
        imgpath_labelpath = self.lines[index].rstrip()
        
        img1, HSVimg, Ycbcrimg, label, depth_label = self._load_data_label(imgpath_labelpath)
        return img1.to("cuda"), HSVimg, Ycbcrimg, label, depth_label

    def _load_data_label(self, imgpath: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Load and process all modalities"""
        img_path1 = imgpath.split(" ")[0]
        label = int(imgpath.split(" ")[1])
        
        img1 = self._load_image(img_path1)
        HSVimg = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        Ycbcrimg = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
        
        depth_label = self._create_depth_label(label)
        
        if self.transform:
            img1 = self.transform(img1)
            HSVimg = self.transform(HSVimg)
            Ycbcrimg = self.transform(Ycbcrimg)
            
        depth_label = self.depth_transform(depth_label)
        return img1, HSVimg, Ycbcrimg, label, depth_label


class TestMultiModalDataset(MultiModalDataset):
    """Test version that also returns image path"""
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, str]:
        imgpath_labelpath = self.lines[index].rstrip()
        img1, HSVimg, Ycbcrimg, label, depth_label = self._load_data_label(imgpath_labelpath)
        imgpath = imgpath_labelpath.split(" ")[0]
        return img1.to("cuda"), HSVimg, Ycbcrimg, label, depth_label, imgpath


class MetaLearningDataset(BaseImageDataset):
    """Dataset for meta-learning with real/attack pairs"""
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        imgpath_labelpath = self.lines[index].rstrip()
        img, meta_real_img, meta_attack_img, label = self._load_data_label(imgpath_labelpath)
        return img.to("cuda"), meta_real_img, meta_attack_img, label

    def _load_data_label(self, imgpath: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        img_path1 = imgpath.split(" ")[0]
        meta_real_img_path = imgpath.split(" ")[1]
        meta_attack_img_path = imgpath.split(" ")[2]
        label = int(imgpath.split(" ")[3])
        
        img1 = self._load_image(img_path1)
        meta_real_img = self._load_image(meta_real_img_path)
        meta_attack_img = self._load_image(meta_attack_img_path)
        
        if self.transform:
            img1 = self.transform(img1)
            meta_real_img = self.transform(meta_real_img)
            meta_attack_img = self.transform(meta_attack_img)
            
        return img1, meta_real_img, meta_attack_img, label


class DepthDataset(MultiModalDataset):
    """Dataset with external depth maps"""
    
    def _load_data_label(self, imgpath: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        depth_label_path = imgpath.split(" ")[0]
        img_path1 = imgpath.split(" ")[1]
        label = int(imgpath.split(" ")[2])
        
        img1 = self._load_image(img_path1)
        HSVimg = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        Ycbcrimg = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
        depth_label = cv2.imread(depth_label_path)
        
        if self.transform:
            img1 = self.transform(img1)
            HSVimg = self.transform(HSVimg)
            Ycbcrimg = self.transform(Ycbcrimg)
            
        depth_label = self.depth_transform(depth_label)
        return img1, HSVimg, Ycbcrimg, label, depth_label


class SURFDataset(MultiModalDataset):
    """Dataset for surface analysis with IR data"""
    
    def _load_data_label(self, imgpath: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        img_path = imgpath.split(" ")[0]
        depth_path = imgpath.split(" ")[1]
        ir_path = imgpath.split(" ")[2]
        label = int(imgpath.split(" ")[3])
        
        img1 = self._load_image(img_path)
        depth = self._load_image(depth_path)
        ir = self._load_image(ir_path)
        
        depth_label = self._create_depth_label(label)
        
        if self.transform:
            img1 = self.transform(img1)
            HSVimg = self.transform(depth)  # Note: Reusing HSVimg name for depth
            Ycbcrimg = self.transform(ir)   # Note: Reusing Ycbcrimg name for IR
            
        depth_label = self.depth_transform(depth_label)
        return img1, HSVimg, Ycbcrimg, label, depth_label
