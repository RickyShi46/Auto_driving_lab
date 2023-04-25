import os
import numpy as np

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from MyPytorchModel import MyPytorchModel, CIFAR10DataModule

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
logger = TensorBoardLogger("tb_logs", name="Resnet50_on_CIFAR10")
hparams = {
    "learning_rate": 1e-4,
    "num_class": 10,
    "loading_method": 'Image',
    "num_workers": 8,
    "batch_size": 128
}
# Set up the data module including your implemented transforms
data_module = CIFAR10DataModule(hparams)
data_module.prepare_data()
# Initialize our model
model = MyPytorchModel(hparams)

trainer = pl.Trainer(
    max_epochs=10,
    logger=logger,
    accelerator="auto"
)

trainer.fit(model, data_module)