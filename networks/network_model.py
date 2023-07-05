
"""
  *  @copyright (c) 2020 Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *  @file    network_model.py
  *  @author  Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *
  *  @brief Class declaration for building the network model.  
 """
 

import torch
import torch.nn as nn
from torchsummary import summary
"""
* @brief Class declaration for building the training network model.
* @param None.
* @return The built netwwork.
"""
#原版
#  class model_cnn(nn.Module):
#     """
#     * @brief Initializes the class varaibles
#     * @param None.
#     * @return None.
#     """
#     def __init__(self):
#         super().__init__()

#         self.elu = nn.ELU()
#         self.dropout = nn.Dropout()

#         self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
#         self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
#         self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2) #384 kernels, size 3x3
#         self.conv_3 = nn.Conv2d(48, 64, kernel_size=3) # 384 kernels size 3x3
#         self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # 256 kernels, size 3x3

#         self.fc0 = nn.Linear(1152, 100)
#         self.fc1 = nn.Linear(100,50)
#         self.fc2 = nn.Linear(50, 10)
#         self.fc3 = nn.Linear(10, 1)
#     """ 
#     * @brief Function to build the model.
#     * @parma The image to train.
#     * @return The trained prediction network.
#     """
#     def forward(self, input):
#         input = input/127.5-1.0
#         input = self.elu(self.conv_0(input))
#         input = self.elu(self.conv_1(input))
#         input = self.elu(self.conv_2(input))
#         input = self.elu(self.conv_3(input))
#         input = self.elu(self.conv_4(input))
#         input = self.dropout(input)

#         input = input.flatten()
#         input = self.elu(self.fc0(input))
#         input = self.elu(self.fc1(input))
#         input = self.elu(self.fc2(input))
#         input = self.fc3(input)

#         return input

# 改动了层数的
# class model_cnn(nn.Module):
#     def __init__(self):
#         super(model_cnn, self).__init__()
#         self.elu = nn.ELU()
#         self.dropout = nn.Dropout(p=0.5)

#         self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2)
#         self.conv2 = nn.Conv2d(24, 48, kernel_size=5, stride=2)
#         self.conv3 = nn.Conv2d(48, 64, kernel_size=5, stride=2)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
#         self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
#         self.dropout2 = nn.Dropout(p=0.5)
#         self.fc1 = nn.Linear(1152, 100)
#         self.fc2 = nn.Linear(100, 50)
#         self.fc3 = nn.Linear(50, 10)
#         self.fc4 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = x / 127.5 - 1.0
#         x = self.elu(self.conv1(x))
#         x = self.elu(self.conv2(x))
#         x = self.elu(self.conv3(x))
#         x = self.elu(self.conv4(x))
#         x = self.elu(self.conv5(x))
#         x = self.dropout2(x)
#         x = x.reshape(x.size(0), -1)

#         x = self.elu(self.fc1(x))
#         x = self.elu(self.fc2(x))
#         x = self.elu(self.fc3(x))
#         x = self.fc4(x)

#         return x

#加了LSTM的
#  class model_cnn(nn.Module):
#     def __init__(self):
#         super(model_cnn, self).__init__()

#         self.elu = nn.ELU()
#         self.dropout = nn.Dropout()

#         self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
#         self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
#         self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
#         self.conv_3 = nn.Conv2d(48, 64, kernel_size=3)
#         self.conv_4 = nn.Conv2d(64, 64, kernel_size=3)
#         self.fc0 = nn.Linear(1152, 100)
#         self.fc1 = nn.Linear(100, 50)
#         self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=2, batch_first=True)
#         self.fc2 = nn.Linear(64, 10)
#         self.fc3 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = x / 127.5 - 1.0
#         x = self.elu(self.conv_0(x))
#         x = self.elu(self.conv_1(x))
#         x = self.elu(self.conv_2(x))
#         x = self.elu(self.conv_3(x))
#         x = self.elu(self.conv_4(x))
#         x = self.dropout(x)

#         x = x.flatten(start_dim=1)  
#         x = self.elu(self.fc0(x))
#         x = self.elu(self.fc1(x))
#         x = x.view(x.size(0), 1, -1)  
#         _, (hn, _) = self.lstm(x)
#         x = hn[-1] 
#         x = self.elu(self.fc2(x))
#         x = self.fc3(x)

#         return x

# #加了batchnorm的
# class model_cnn(nn.Module):

#     def __init__(self):
#         super(model_cnn, self).__init__()
#         self.elu = nn.ELU()
#         self.dropout = nn.Dropout()

#         self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
#         self.bn_0 = nn.BatchNorm2d(24)
#         self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
#         self.bn_1 = nn.BatchNorm2d(36)
#         self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2) #384 kernels, size 3x3
#         self.bn_2 = nn.BatchNorm2d(48)
#         self.conv_3 = nn.Conv2d(48, 64, kernel_size=3) # 384 kernels size 3x3
#         self.bn_3 = nn.BatchNorm2d(64)
#         self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # 256 kernels, size 3x3
#         self.bn_4 = nn.BatchNorm2d(64)

#         self.fc0 = nn.Linear(1152, 100)
#         self.fc1 = nn.Linear(100,50)
#         self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=2, batch_first=True)
#         self.fc2 = nn.Linear(64, 10)
#         self.fc3 = nn.Linear(10, 1)

#     def forward(self, input):
#         input = input/127.5-1.0
#         input = self.elu(self.bn_0(self.conv_0(input)))
#         input = self.elu(self.bn_1(self.conv_1(input)))
#         input = self.elu(self.bn_2(self.conv_2(input)))
#         input = self.elu(self.bn_3(self.conv_3(input)))
#         input = self.elu(self.bn_4(self.conv_4(input)))
#         input = self.dropout(input)

#         input = input.flatten(start_dim=1)
#         input = self.elu(self.fc0(input))
#         input = self.elu(self.fc1(input))
#         input = input.view(input.size(0), 1, -1)  
#         _, (hn, _) = self.lstm(input)
#         input = hn[-1] 
#         input = self.elu(self.fc2(input))
#         input = self.fc3(input)

#         return input






# #加了自注意力机制
# class SelfAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(SelfAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.projection = nn.Sequential(
#             nn.Linear(hidden_size, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 1)
#         )

#     def forward(self, encoder_outputs):
#         # encoder_outputs的形状: [batch_size, sequence_length, hidden_size]
#         energy = self.projection(encoder_outputs)  # 计算能量分数
#         weights = torch.softmax(energy.squeeze(-1), dim=1)  # 对能量进行softmax得到权重
#         outputs = (encoder_outputs * weights.unsqueeze(2)).sum(dim=1)  # 对encoder_outputs加权求和
#         return outputs, weights
# class model_cnn(nn.Module):
#     def __init__(self):
#         super(model_cnn, self).__init__()
#         self.elu = nn.ELU()
#         self.dropout = nn.Dropout()

#         self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
#         self.bn_0 = nn.BatchNorm2d(24)
#         self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
#         self.bn_1 = nn.BatchNorm2d(36)
#         self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
#         self.bn_2 = nn.BatchNorm2d(48)
#         self.conv_3 = nn.Conv2d(48, 64, kernel_size=3)
#         self.bn_3 = nn.BatchNorm2d(64)
#         self.conv_4 = nn.Conv2d(64, 64, kernel_size=3)
#         self.bn_4 = nn.BatchNorm2d(64)

#         self.fc0 = nn.Linear(1152, 100)
#         self.fc1 = nn.Linear(100, 50)
#         self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=2, batch_first=True)
#         self.self_attention = SelfAttention(64)  # 添加SelfAttention层
#         self.fc2 = nn.Linear(64, 10)
#         self.fc3 = nn.Linear(10, 1)
#     def forward(self, input):
#         input = input / 127.5 - 1.0
#         input = self.elu(self.bn_0(self.conv_0(input)))
#         input = self.elu(self.bn_1(self.conv_1(input)))
#         input = self.elu(self.bn_2(self.conv_2(input)))
#         input = self.elu(self.bn_3(self.conv_3(input)))
#         input = self.elu(self.bn_4(self.conv_4(input)))
#         input = self.dropout(input)

#         input = input.flatten(start_dim=1)
#         input = self.elu(self.fc0(input))
#         input = self.elu(self.fc1(input))
#         input = input.view(input.size(0), 1, -1)
#         _, (hn, _) = self.lstm(input)
#         input, _ = self.self_attention(hn[-1].unsqueeze(0))  # 应用SelfAttention层并增加维度
#         input = self.elu(self.fc2(input))
#         input = self.fc3(input)

#         return input


# #加了防过拟合
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)  
        weights = F.softmax(energy.squeeze(-1), dim=1)  
        weights = self.dropout(weights)  
        outputs = (encoder_outputs * weights.unsqueeze(2)).sum(dim=1)  
        return outputs, weights

class model_cnn(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(model_cnn, self).__init__()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)

        self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
        self.bn_0 = nn.BatchNorm2d(24)
        self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.bn_1 = nn.BatchNorm2d(36)
        self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.bn_2 = nn.BatchNorm2d(48)
        self.conv_3 = nn.Conv2d(48, 64, kernel_size=3)
        self.bn_3 = nn.BatchNorm2d(64)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn_4 = nn.BatchNorm2d(64)

        self.fc0 = nn.Linear(1152, 100)
        self.fc1 = nn.Linear(100, 50)
        self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=2, batch_first=True)
        self.self_attention = SelfAttention(64, dropout_rate)  
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, input):
        input = input / 127.5 - 1.0
        input = self.elu(self.bn_0(self.conv_0(input)))
        input = self.elu(self.bn_1(self.conv_1(input)))
        input = self.elu(self.bn_2(self.conv_2(input)))
        input = self.elu(self.bn_3(self.conv_3(input)))
        input = self.elu(self.bn_4(self.conv_4(input)))
        input = self.dropout(input)

        input = input.flatten(start_dim=1)
        input = self.elu(self.fc0(input))
        input = self.elu(self.fc1(input))
        input = input.view(input.size(0), 1, -1)
        _, (hn, _) = self.lstm(input)
        input, _ = self.self_attention(hn[-1].unsqueeze(0)) 
        input = self.elu(self.fc2(input))
        input = self.dropout(input)  
        input = self.fc3(input)

        return input

#多注意头
# import torch.nn.functional as F


# class SelfAttention(nn.Module):
#     def __init__(self, hidden_size, num_heads, dropout_rate):
#         super(SelfAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.dropout = nn.Dropout(dropout_rate)
        
#         self.projection = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(True),
#             nn.Linear(hidden_size, num_heads * hidden_size)
#         )

#     def forward(self, encoder_outputs):
#         energy = self.projection(encoder_outputs)  
#         energy = energy.view(-1, self.num_heads, self.hidden_size).transpose(1, 2)
#         weights = F.softmax(energy.squeeze(), dim=1) 
#         weights = self.dropout(weights)  
#         outputs = torch.matmul(weights.transpose(-1, -2), encoder_outputs.transpose(1, 2)).transpose(1, 2)

#         outputs = outputs.sum(dim=1)
#         return outputs, weights

# class model_cnn(nn.Module):
#     def __init__(self, dropout_rate=0.5):
#         super(model_cnn, self).__init__()
#         self.elu = nn.ELU()
#         self.dropout = nn.Dropout(dropout_rate)

#         self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
#         self.bn_0 = nn.BatchNorm2d(24)
#         self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
#         self.bn_1 = nn.BatchNorm2d(36)
#         self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
#         self.bn_2 = nn.BatchNorm2d(48)
#         self.conv_3 = nn.Conv2d(48, 64, kernel_size=3)
#         self.bn_3 = nn.BatchNorm2d(64)
#         self.conv_4 = nn.Conv2d(64, 64, kernel_size=3)
#         self.bn_4 = nn.BatchNorm2d(64)

#         self.fc0 = nn.Linear(1152, 100)
#         self.fc1 = nn.Linear(100, 50)
#         self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=2, batch_first=True)
#         self.self_attention = SelfAttention(64, num_heads=8, dropout_rate=dropout_rate)  
#         self.fc2 = nn.Linear(8, 10)
#         self.fc3 = nn.Linear(10, 1)

#     def forward(self, input):
#         input = input / 127.5 - 1.0
#         input = self.elu(self.bn_0(self.conv_0(input)))
#         input = self.elu(self.bn_1(self.conv_1(input)))
#         input = self.elu(self.bn_2(self.conv_2(input)))
#         input = self.elu(self.bn_3(self.conv_3(input)))
#         input = self.elu(self.bn_4(self.conv_4(input)))
#         input = self.dropout(input)

#         input = input.flatten(start_dim=1)
#         input = self.elu(self.fc0(input))
#         input = self.elu(self.fc1(input))
#         input = input.view(input.size(0), 1, -1)
#         _, (hn, _) = self.lstm(input)
#         input, _ = self.self_attention(hn[-1].unsqueeze(0)) 
#         input = self.elu(self.fc2(input))
#         input = self.dropout(input)  
#         input = self.fc3(input)
#         return input
# class model_cnn(nn.Module):
#     def __init__(self):
#             super().__init__()

#             self.model_layers = nn.Sequential(
#                 nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
#                 nn.ELU(),
#                 nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
#                 nn.ELU(),

#                 nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
#                 nn.BatchNorm2d(48),
#                 nn.ELU(),

#                 nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3),
#                 nn.BatchNorm2d(64),
#                 nn.ELU(),

#                 nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
#                 nn.BatchNorm2d(64),
#                 nn.ELU(),

#                 # kernel size is 9 - to reduce spatial dimension to 1x1 from 9x9
#                 nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9),
#                 nn.ELU(),

#                 nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
#                 nn.ELU(),

#                 nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
#                 nn.Tanh()
#             )

#     def forward(self, x):
#         x = self.model_layers(x)
#         x = torch.squeeze(x)
#         return x



# import torchvision.models as models
# class model_cnn(nn.Module):
#         def __init__(self):
#             super().__init__()

#             self.resnet50 = models.resnet50(pretrained=True)
#             self.lstm = nn.LSTM(input_size=1000, hidden_size=128, num_layers=1, batch_first=True)
#             self.fc = nn.Linear(128, 1)  # 修改输出层为1个节点
#             #self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)


#         def forward(self, input):
#             input = input / 127.5 - 1.0
#             features = self.resnet50(input)
            
#              # 将ResNet50的输出作为LSTM的输入
#             lstm_output, _ = self.lstm(features.unsqueeze(0))

#             # 取最后一个时间步的输出作为预测结果
#             output = self.fc(lstm_output[:, -1, :])
            

#             return output


# import torchvision.models as models
# class model_cnn(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.resnet50 = models.resnet50(pretrained=True)
#         self.lstm = nn.LSTM(input_size=1000, hidden_size=128, num_layers=1, batch_first=True)
#         self.fc = nn.Linear(128, 1)  # 修改输出层为1个节点

#     def forward(self, input):
#         input = input / 127.5 - 1.0
#         batch_size, num_frames, channels, height, width = input.size()
#         input = input.view(batch_size * num_frames, channels, height, width)

#         features = self.resnet50(input)

#         features = features.view(batch_size, num_frames, -1)
#         lstm_output, _ = self.lstm(features)

#         output = self.fc(lstm_output[:, -1, :])

#         return output


# import torchvision.models as models

# class model_cnn(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.resnet50 = models.resnet50(pretrained=True)
#         self.lstm = nn.LSTM(input_size=1000, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(256, 1)

#     def forward(self, input):
#         input = input / 127.5 - 1.0
#         batch_size, num_frames, channels, height, width = input.size()

#         features = self.resnet50(input.view(batch_size * num_frames, channels, height, width))

#         features = features.view(batch_size, num_frames, -1)
#         lstm_output, _ = self.lstm(features)

#         output = self.fc(lstm_output[:, -1, :])

#         return output


