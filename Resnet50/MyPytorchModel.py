import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm


class ResNet50BasicBlock(pl.LightningModule):
    def __init__(self, in_channel, outs, kernerl_size, stride, padding):
        super(ResNet50BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernerl_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernerl_size[1], stride=stride[0], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernerl_size[2], stride=stride[0], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(outs[2])

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        return F.relu(out + x)


class ResNet50DownBlock(pl.LightningModule):
    def __init__(self, in_channel, outs, kernel_size, stride, padding):
        super(ResNet50DownBlock, self).__init__()
        # out1, out2, out3 = outs
        # print(outs)
        self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(outs[2])

        self.extra = nn.Sequential(
            nn.Conv2d(in_channel, outs[2], kernel_size=1, stride=stride[3], padding=0),
            nn.BatchNorm2d(outs[2])
        )

    def forward(self, x):
        x_shortcut = self.extra(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return F.relu(x_shortcut + out)


class ResNet50(pl.LightningModule):
    def __init__(self, hparams):
        super(ResNet50, self).__init__()
        self.save_hyperparameters(hparams)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResNet50DownBlock(64, outs=[64, 64, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        )

        self.layer2 = nn.Sequential(
            ResNet50DownBlock(256, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50DownBlock(512, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
        )

        self.layer3 = nn.Sequential(
            ResNet50DownBlock(512, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2],
                              padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0])
        )

        self.layer4 = nn.Sequential(
            ResNet50DownBlock(1024, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0])
        )

        #self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, ceil_mode=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(2048, 10)
        #self.conv11 = nn.Conv2d(2048, 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        #out = self.fc(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch
        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        n_total = len(targets)
        return loss, n_correct, n_total
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        length = sum([x[mode + '_n_total'] for x in outputs])
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / length
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "train")
        self.log('loss',loss)

        return {'loss': loss, 'train_n_correct':n_correct, 'train_n_total': n_total}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "val")
        self.log('val_loss',loss)
        return {'val_loss': loss, 'val_n_correct':n_correct, 'val_n_total': n_total}
    
    def test_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct':n_correct, 'test_n_total': n_total}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        self.log('val_loss',avg_loss)
        self.log('val_acc',acc)
        return {'val_loss': avg_loss, 'val_acc': acc}

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), self.hparams["learning_rate"])
        return optim

    def getTestAcc(self, loader):
        self.eval()
        self.to(self.device)

        scores = []
        labels = []

        for batch in tqdm(loader):
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.opt = hparams
        if 'loading_method' not in hparams.keys():
            self.opt['loading_method'] = 'Image'
        if 'num_workers' not in hparams.keys():
            self.opt['num_workers'] = 2

    def prepare_data(self, stage=None, CIFAR_ROOT="../datasets/cifar10"):

        # create dataset
        CIFAR_ROOT = "../datasets/cifar10"
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        my_transform = transforms.Compose([
            transforms.RandomApply((transforms.RandomHorizontalFlip(p=0.8),
                                    transforms.RandomResizedCrop((32,32))),p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Make sure to use a consistent transform for validation/test
        train_val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        # Note: you can change the splits if you want :)
        split = {
            'train': 0.6,
            'val': 0.2,
            'test': 0.2
        }
        split_values = [v for k,v in split.items()]
        assert sum(split_values) == 1.0
        
        if self.opt['loading_method'] == 'Image':
            # Set up a full dataset with the two respective transforms
            cifar_complete_augmented = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=my_transform)
            cifar_complete_train_val = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=train_val_transform)

            N = len(cifar_complete_augmented)

            num_train, num_val, num_test = int(N*split['train']), int(N*split['val']), int(N*split['test'])
            self.train_dataset, _, _= \
                random_split(cifar_complete_augmented,[num_train, num_val, num_test])
            _,self.val_dataset,self.test_dataset = \
                random_split(cifar_complete_train_val,[num_train, num_val, num_test])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.opt['batch_size'],shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.opt['batch_size'],shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.opt['batch_size'])
