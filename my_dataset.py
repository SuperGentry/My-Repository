import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random
from cv2 import cv2

class Data_myself(Dataset):

    def __init__(self, listroot=None, labelroot=None, shuffle=True, transform=None):
        self.listroot = listroot
        self.labelroot = labelroot
        self.transform = transform
        self.transform1 = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Grayscale(num_output_channels=1),
                                    ])
        listfile_root = self.listroot

        with open(listfile_root, 'r') as file:
            self.lines = file.readlines()
        if shuffle:
            random.shuffle(self.lines)
        self.nSamples = len(self.lines)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        imgpath_labelpath = self.lines[index].rstrip()
        img1, label, depth_label= self.load_data_label(imgpath_labelpath)
        img1 = img1.to("cuda")
        return (img1, label, depth_label)

    def load_data_label(self, imgpath):
        img_path1 = imgpath.split(" ")[0]
        label = int(imgpath.split(" ")[1])
        img1 = cv2.imread(img_path1)
        if label == 1:
            depth_label = np.ones((48, 48, 3))
            depth_label = depth_label
            depth_label = depth_label.astype(np.float32)
        else:
            depth_label = np.zeros((48, 48, 3))
            depth_label = depth_label.astype(np.float32)
        if self.transform is not None:
            img1 = self.transform(img1)
        depth_label = self.transform1(depth_label)
        return img1, label, depth_label

