# Train a ResNet50 on Cifar-10

Contributed by Yujie Zhang and Yu Shi

## Introduction
- [x] **MyPytorchModel.py:**
    - Class `ResNet50` Constructs the Resnet50 model with five convolutional layer blocks average pool layer and a fully connected layer.
    - Functions `training_step` and `validation_step` that take a batch as input and calculate the loss.
    - Function `getTestAcc` compares the predictions and labels in batch wise, then calculate the average accuracy.
    - Class `CIFAR10DataModule` deals with the dataset and the dataloader.
    - Function `prepare_data` sets up the dataset and the related transforms for it.
- [x] **main:**
    - Initialize the model and the data with a set of hyperparameters given in the dictionary hparams.
    - Fit model to the trainer.
    - Save the parameters of the model to `Resnet50.pth`.
    - Pass test dataloader to the function `getTestAcc` to get the accuracy and print it out.


## Test
We stored Cifar-10 in the folder `../datasets/cifar10`. We accessed the images in the folder when we built our dataset and dataloader.The input size of each image is tensor(3, 32, 32).

We divided the Cifar-10 dataset into three parts: 
- 60% for training 
- 20% for validation
- 20% for test

We did data augmentation for training and validation dataset, without doing it for test dataset.

We provided a way to test our model on colab:
- Create a new notebook and mount your Google Drive
```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```
- Clone the repository
```
!git clone https://github.com/RickyShi46/Auto_driving_lab.git
```
- Go into the directory
```
cd Auto_driving_lab/Resnet50
```
- Set up PyTorch environment in colab
```
!python -m pip install pytorch-lightning==1.6.0
!pip install tensorrt
```
- Execute the code
```
!python main.py
```
At the end of the code, the test accuracy will be printed out.

## Best result

We achieved test accuracy 92.8% by using learning rate 1e-4, batch size 32 and 77 epochs. We saved our best model parameters in the file `Resnet50.pth`.

