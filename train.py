import argparse
from pathlib import Path
import os
import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torchvision.transforms as transforms 
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import AdaIN_net as net
import datetime as time
import matplotlib.pyplot as plt
from tqdm import tqdm

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True



# Transforms
def train_transform():
     transform_list = [
          transforms.Resize(size=(512, 512)),
          transforms.RandomCrop(256),
          transforms.ToTensor()
     ]
     return transforms.Compose(transform_list)

def change_learning_rate(optimizer, iter_count):
     lr = learnRate / (1.0 + learnRateDecay * iter_count)
     for param_group in optimizer.param_groups:
          param_group['lr'] = lr

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'
    


parser = argparse.ArgumentParser()
parser.add_argument('-content_dir', type=str, help='Dir path to content dataset')
parser.add_argument('-style_dir', type=str, help='Dir path to style dataset')
parser.add_argument('-gamma', type=float, help='gamma value')
parser.add_argument('-e', type=int, help='epoch value')
parser.add_argument('-b', type=int, help='batch value')
parser.add_argument('-l', type=str, default=1.0, help='encoder file path')
parser.add_argument('-s', type=str, help='Save decoder path')
parser.add_argument('-p', type=str, help='decoder image')
parser.add_argument('-cuda', type=str, help='[y/N]')

opt = parser.parse_args()
device = torch.device('cuda:0')

weight = opt.gamma
n_epochs = opt.e
n_batches = opt.b
encoder_file = opt.l
# decoder_png = Image.open(opt.p)

use_cuda = False
if opt.cuda == 'y' or opt.cuda == 'Y':
	use_cuda = True
out_dir = './output/'
os.makedirs(out_dir, exist_ok=True)


decoder = net.encoder_decoder.decoder
encoder = net.encoder_decoder.encoder

encoder.load_state_dict(torch.load(opt.l, map_location=device))

network = net.AdaIN_net(encoder, decoder)
network.train()
network.to(device)

# Content Data
content_transform = train_transform()
style_transform = train_transform()

content_dataset = FlatFolderDataset(opt.content_dir, content_transform)
content_dataloader = data.DataLoader(content_dataset, batch_size=n_batches, shuffle=True)

# Style Data

style_dataset = FlatFolderDataset(opt.style_dir, style_transform)
style_dataloader = data.DataLoader(style_dataset, batch_size=n_batches, shuffle=True)

learnRate = 0.0001
learnRateDecay = 0.00001
optimizer = torch.optim.Adam(network.decoder.parameters(), lr=learnRate)

lossContent = []
lossStyle = []
lossContSty = []

for epoch in tqdm(range(n_epochs)):
    loss_batch_c = 0.0
    loss_batch_s = 0.0
    loss_batch_cs = 0.0
    change_learning_rate(optimizer, iter_count=epoch)
    for batch in range(n_batches):
        torch.cuda.empty_cache()
        
        content_images = next(iter(content_dataloader)).to(device)
        style_images = next(iter(style_dataloader)).to(device)
        
        loss_c, loss_s = network(content_images, style_images)
        loss = loss_c + weight * loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_batch_c += loss_c.item()
            loss_batch_s += loss_s.item()
            loss_batch_cs += loss.item()
    
    avg_lossContent = loss_batch_c / n_batches
    avg_lossStyle = loss_batch_s / n_batches
    avg_lossContSty = loss_batch_cs / n_batches


    lossContent.append(avg_lossContent)
    lossStyle.append(avg_lossStyle)
    lossContSty.append(avg_lossContSty)
    
    if (epoch + 1) == opt.e:
        state_dict = net.encoder_decoder.decoder.state_dict()

        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device(device))

        torch.save(state_dict, opt.s.format(epoch + 1))

# torch.save(network.state_dict(), opt.s)
plt.figure(2, figsize=(12, 17))
plt.clf()
plt.plot(lossContSty, label='content+style')
plt.plot(lossContent, label='content')
plt.plot(lossStyle, label='style')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc=1)
plt.savefig(opt.p)