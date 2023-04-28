import os
import numpy as np

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from MyPytorchModel import ResNet50, CIFAR10DataModule

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.empty_cache()
    logger = TensorBoardLogger("tb_logs", name="Resnet50_on_CIFAR10")
    hparams = {
        "learning_rate": 1e-3,
        "num_class": 10,
        "loading_method": 'Image',
        "num_workers": 8,
        "batch_size": 32
    }
    # Set up the data module including your implemented transforms
    data_module = CIFAR10DataModule(hparams)
    data_module.prepare_data()
    # Initialize our model
    model = ResNet50(hparams)

    trainer = pl.Trainer(
        max_epochs=10,
        logger=logger,
        accelerator="auto"
    )

    trainer.fit(model, data_module)
    _, test_acc = model.getTestAcc(data_module.test_dataloader())
    print(f"Test Accuracy : {test_acc*100}%")

if __name__ == '__main__':
    main()
