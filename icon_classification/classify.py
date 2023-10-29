# License: BSD
# Author: Sasank Chilamkurthy
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import natsort

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = list(filter(lambda x: x[-12:] =="_overlap.png", os.listdir(main_dir)))
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

cudnn.benchmark = True

def left_crop(image):
    w, h = transforms.functional.get_image_size(image)
    return transforms.functional.crop(image, 0, w//2, h, w-(w//2))

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'test': transforms.Compose([
        transforms.Lambda(left_crop), #experiment with removing left crop
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'ICON/train_images_true/icon-filter/png'
image_datasets = {x: CustomDataSet(data_dir,
                                          data_transforms[x])
                  for x in ['test']}
dataloaders = ({x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=False, num_workers=4)
              for x in ['test']})
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = ['bad', 'good']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_model(model):
    was_training = model.training
    model.eval()
    images_so_far = 0
    with open('labeled_reconstructions.txt', 'w') as writefile:
      with torch.no_grad():
          imgs = image_datasets['test'].total_imgs
          for i, (inputs) in enumerate(dataloaders['test']):
              inputs = inputs.to(device)

              outputs = model(inputs)
              _, preds = torch.max(outputs, 1)
              for j in range(inputs.size()[0]):
                  root = imgs[images_so_far]
                  label = class_names[preds[j]]
                  img_csv = root+","+label+"\n"
                  writefile.write(img_csv)

                  images_so_far += 1

          model.train(mode=was_training)
        
model = torch.load('model_conv.pt')
test_model(model)