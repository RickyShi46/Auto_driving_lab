"""
  *  @copyright (c) 2020 Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *  @file    train.py
  *  @author  Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *
  *  @brief Main file to train and evaluate the model.  
 """
 #主要功能是在给定的数据集上训练自动驾驶模型，并在训练过程中对模型进行验证。
import os
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from networks.network_model import model_cnn
from data_extractor import Features
import utils
import argparse
from matplotlib import pyplot as plt
import cv2
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

# Global declaration of the arrays to plot the graphs.
loss_vals = []
train_step = []
val_step = []
val_losses = []



"""
* @brief Function to train the model with the input data and save them.
* @param The arguments containing the parameters 
*  needed to train and generate the model.
* @param The model to train the data with.
* @param The split datatset to train.
* @param The validation dataset.
* @return None.
"""
#train_model(args, model, dataset_train, dataset_val)：该函数是用于训练自动驾驶模型的函数。其中 args 是训练所需的参数，model 是训练的模型，dataset_train 是用于训练的数据集，dataset_val 是用于验证的数据集。该函数通过读取数据集中的图像和标签进行训练，并使用 Adam 优化器和均方误差（MSE）损失函数进行优化。在训练过程中，会对每个 epoch 进行采样，并对数据进行数据增强（如随机翻转、随机移动、随机阴影、随机亮度等），然后计算损失值，并通过反向传播更新模型参数。在每个 epoch 的末尾，会对模型进行验证，并计算验证损失。最后，该函数会将模型保存到文件中，以备后续使用。
def train_model(args, model, dataset_train, dataset_val):
    # Imports the training model.
    model.train()
    #Declaration of the optimizer and the loss model.

    #optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    criterion = nn.MSELoss()
    step = 0 # steps initialization
    imgs_per_batch = args.batch_size #gets the batch size from the argument parameters
    optimizer.zero_grad()
    for epoch in range(args.nb_epoch): # runs for the number of eposchs set in the arguments
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=args.samples_per_epoch)
        # scheduler.step()
        for i, sample_id in enumerate(sampler):
            data = dataset_train[sample_id]
            label = data['steering_angle'] #, data['brake'], data['speed'], data['throttle']
            img_pth, label = utils.choose_image(label)
			# Data augmentation and processing steps   
            img = cv2.imread(data[img_pth].strip())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = utils.preprocess(img)

            # img = utils.random_rotate(img)
            # img = utils.random_scale(img)
            # img = utils.random_crop(img)

            img, label = utils.random_flip(img, label)
            img, label = utils.random_translate(img, label, 100, 10)
            img = utils.random_shadow(img)
            img = utils.random_brightness(img)
            img = Variable(torch.cuda.FloatTensor(np.array([img])))
            label = np.array([label]).astype(float)
            
           # label = np.expand_dims(label, axis=0) 
            
            label = Variable(torch.cuda.FloatTensor(label))
            img = img.permute(0,3,1,2)
            #print(img.shape)
            #training and loss calculation
            out_vec = model(img)
            loss = criterion(out_vec,label)

            loss.backward()
            if step%imgs_per_batch==0:
                optimizer.step()
                optimizer.zero_grad()

            # Status update for the network. To see the working
            if step%20==0:
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    'Train Loss: {:.8f} |'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    loss.item())
                train_step.append(step)
                loss_vals.append(loss.item())
                #Uncomment the line below if you want to see the working
                # print(log_str)

            if step%100==0:
                log_value('train_loss',loss.item(),step)
            
            if step%5000==0:
                # Validation of the model mid training for better understanding and visualization
                val_loss = eval_model(model,dataset_val, num_samples=1470)
                log_value('val_loss',val_loss,step)
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | Val Loss: {:.8f}'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    val_loss)
                val_losses.append(val_loss)
                val_step.append(step)
                print(log_str)
                model.train()  # resumes the training process

            if step%5000==0:
            	# Saves the intermediate points in the training process for testing in simulator.
                if not os.path.exists(args.model_dir):
                    os.makedirs(args.model_dir)

                reflex_pth = os.path.join(
                    args.model_dir,
                    'model_{}'.format(step))
                torch.save(
                    model.state_dict(),
                    reflex_pth)

            step += 1
       

"""
* @brief Function to evaluate the model generated by the training process
* @param Model to be evaluated
* @param The validation dataset
* @param the sample size to evaluate.
* @return The validarion loss.
""" 
#train_model(args, model, dataset_train, dataset_val)：该函数是用于训练自动驾驶模型的函数。其中 args 是训练所需的参数，model 是训练的模型，dataset_train 是用于训练的数据集，dataset_val 是用于验证的数据集。该函数通过读取数据集中的图像和标签进行训练，并使用 Adam 优化器和均方误差（MSE）损失函数进行优化。在训练过程中，会对每个 epoch 进行采样，并对数据进行数据增强（如随机翻转、随机移动、随机阴影、随机亮度等），然后计算损失值，并通过反向传播更新模型参数。在每个 epoch 的末尾，会对模型进行验证，并计算验证损失。最后，该函数会将模型保存到文件中，以备后续使用。
def eval_model(model,dataset,num_samples):
    model.eval()
    criterion = nn.MSELoss()
    step = 0
    val_loss = 0
    count = 0
    sampler = RandomSampler(dataset)
    torch.manual_seed(0)
    for sample_id in tqdm(sampler):
        if step==num_samples:
            break

        data = dataset[sample_id]
        img_pth, label = utils.choose_image(data['steering_angle'])
        # image preprocessing and augmentation.
        img = cv2.imread(data[img_pth].strip())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img = utils.random_rotate(img)
        # img = utils.random_scale(img)
        # img = utils.random_crop(img)
        
        img = utils.preprocess(img)
        img, label = utils.random_flip(img, label)
        img, label = utils.random_translate(img, label, 100, 10)
        img = utils.random_shadow(img)
        img = utils.random_brightness(img)
        img = Variable(torch.cuda.FloatTensor(np.array([img])))
        img = img.permute(0,3,1,2)
        label = np.array([label]).astype(float)

       # label = np.expand_dims(label, axis=0) 

        label = Variable(torch.cuda.FloatTensor(label))

        out_vec = model(img)

        loss = criterion(out_vec,label)

        batch_size = 4
        val_loss += loss.data.item()
        count += batch_size
        step += 1

    val_loss = val_loss / float(count)
    return val_loss

"""
* @brief The main file to run the training of the dataset and generate the model.
* @param The arguments with the training parameters.
* @return None.
"""  
def main(args):
	#build and import the network model.
    model = model_cnn()
    #Check for cuda availability
    if torch.cuda.is_available():
        model = model.cuda()


    print('Creating model ...')
    #model = Reflex_CNN().cuda()
    configure("log/")
    print('Creating data loaders ...')
    dataset = Features(args.data_dir)
    train_size = int(args.train_size * len(dataset))
    test_size = len(dataset) - train_size
    dataset_train, dataset_val = torch.utils.data.dataset.random_split(dataset,[train_size, test_size])#随机选择数据集


    ## 按照时间顺序划分数据集
    # dataset_train = torch.utils.data.Subset(dataset, range(train_size))
    # dataset_val = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))



    train_model(args, model,dataset_train, dataset_val)
    # plot the loss graphs from the training file.
    plt.plot(train_step,loss_vals)
    plt.xlabel("Train Steps")
    plt.ylabel("Train Loss")
    plt.savefig('train_loss(%d,%f,%f).png'%(args.nb_epoch,args.learning_rate,args.keep_prob))
    plt.clf()
    # plt.show()
    # plor the loss graphs from the validation step
    plt.plot(val_step,val_losses)
    plt.xlabel("Validation Steps")
    plt.ylabel("Validation Loss")
    plt.savefig('val_loss(%d,%f,%f).png'%(args.nb_epoch,args.learning_rate,args.keep_prob))
    # plt.show()
    plt.clf()

"""
* @brief Runs the main function and gets the arguments from the user or 
* takes in the default set values
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-m', help='model directory',       dest='model_dir',         type=str,   default='models_resnet')
    parser.add_argument('-t', help='train size fraction',   dest='train_size',        type=float, default=0.8)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=2)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)

    # parser.add_argument('--step_size', type=int, default=10, help='Step size for learning rate scheduler')
    # parser.add_argument('--gamma', type=float, default=0.1, help='Gamma value for learning rate scheduler')

    args = parser.parse_args()

    main(args)
