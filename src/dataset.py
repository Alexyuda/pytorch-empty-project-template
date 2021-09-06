from torch.utils.data import Dataset
from glob import glob
import os
import imagesize
from utils import *
import math
from PIL import Image
import random
import torchvision.transforms as transforms
from torchvision.transforms import functional as tvF
import numpy as np
import torch
import cv2


class DummyDataset(Dataset):

    def __init__(self, args):
     
    def __len__(self):
        return bla.__len__()

    def __getitem__(self, i):
        return datum
