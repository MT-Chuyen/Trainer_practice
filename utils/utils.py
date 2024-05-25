import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms import functional as F
import torchvision
import torch.utils.tensorboard as tb
from torchvision import transforms, datasets, models
from skimage.segmentation import find_boundaries as fb
 
import numpy as np
import csv
#### change to different name
SELECT_LABEL_NAMES = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 
                     'pole', 'traffic light','traffic sign', 'vegetation', 'terrain','sky','person','rider','car','truck','bus',
                     'train','motorcycle','bicycle']

N_CLASSES = len(SELECT_LABEL_NAMES)

#print("Number of classes", N_CLASSES)

### Weights for Focal loss
FOCAL_LOSS_WEIGHTS = [0.0, 0.1825, 0.0525, 0.0525, 0.0525, 0.0525, 0.025, 0.01, 0.01, 0.025, 0.0525, 0.1, 
                            0.0525, 0.0525, 0.08, 0.04, 0.04, 0.04, 0.04, 0.04]
class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=N_CLASSES):
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        self.matrix = self.matrix.to(preds.device) ###### issue here...
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)
