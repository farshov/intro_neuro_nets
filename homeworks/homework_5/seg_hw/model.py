import torch.nn as nn


def conv_bn_relu(in_planes, out_planes, kernel=3, stride=1, padding=1):
     net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
                         nn.BatchNorm2d(num_features=out_planes),
                         nn.ReLU())
     return net;

# input size Bx3x224x224
class SegmenterModel(nn.Module):
    def __init__(self, in_size=3):
        super(SegmenterModel, self).__init__()
        self.pool_idx = [0] * 5
        self.decoders = []
        self.encoders = []
        self.pool = []
        self.unpool = []
        self.num_classes = 2
        #__________________________________________________
        # Encoder part
        # input 3x224x224
        self.encoder_1 = nn.Sequential()
        self.encoder_1.add_module('conv_1_1', conv_bn_relu(3, 64, kernel=3, stride=1, padding=1))
        self.encoder_1.add_module('conv_1_2',conv_bn_relu(64, 128, kernel=3, stride=1, padding=1))
        self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        self.encoders.append(self.encoder_1)
        # output 128x112x112 

        # input 128x112x112
        self.encoder_2 = nn.Sequential()
        self.encoder_2.add_module('conv_2_1', conv_bn_relu(128, 128, kernel=3, stride=1, padding=1))
        self.encoder_2.add_module('conv_2_2',conv_bn_relu(128, 256, kernel=3, stride=1, padding=1))
        self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        self.encoders.append(self.encoder_2)
        # output 256x56x56

        # input 256x56x56
        self.encoder_3 = nn.Sequential()
        self.encoder_3.add_module('conv_3_1', conv_bn_relu(256, 256, kernel=3, stride=1, padding=1))
        #self.encoder_3.add_module('conv_3_2',conv_bn_relu(256, 256, kernel=3, stride=1, padding=1))
        self.encoder_3.add_module('conv_3_3',conv_bn_relu(256, 512, kernel=3, stride=1, padding=1))
        self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        self.encoders.append(self.encoder_3)
        # output 512x28x28

        # input 512x28x28
        self.encoder_4 = nn.Sequential()
        self.encoder_4.add_module('conv_4_1', conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        self.encoder_4.add_module('conv_4_2',conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        #self.encoder_4.add_module('conv_4_3',conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        self.encoders.append(self.encoder_4)
        # output 512x14x14

        # input 512x14X14
        self.encoder_5 = nn.Sequential()
        self.encoder_5.add_module('conv_5_1', conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        self.encoder_5.add_module('conv_5_2',conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        #self.encoder_5.add_module('conv_5_3',conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        self.encoders.append(self.encoder_5)
        # output 512x7x7
        #__________________________________________________
        
        #__________________________________________________
        # Decoder part
        # input 512x7x7
        self.decoder_1 = nn.Sequential()
        self.unpool.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_1.add_module('conv_5_1', conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        self.decoder_1.add_module('conv_5_2',conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        #self.decoder_1.add_module('conv_5_3',conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        self.decoders.append(self.decoder_1)
        # output 512x14x14

        # input 512x14x14
        self.decoder_2 = nn.Sequential()
        self.unpool.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_2.add_module('conv_4_1', conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        self.decoder_2.add_module('conv_4_2',conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        #self.decoder_2.add_module('conv_4_3',conv_bn_relu(512, 512, kernel=3, stride=1, padding=1))
        self.decoders.append(self.decoder_2)
        # output 512x28x28

        # input 512x28x28
        self.decoder_3 = nn.Sequential()
        self.unpool.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_3.add_module('conv_3_1', conv_bn_relu(512, 256, kernel=3, stride=1, padding=1))
        self.decoder_3.add_module('conv_3_2',conv_bn_relu(256, 256, kernel=3, stride=1, padding=1))
        #self.decoder_3.add_module('conv_3_3',conv_bn_relu(256, 256, kernel=3, stride=1, padding=1))
        self.decoders.append(self.decoder_3)
        # output 256x56x56

        # input 256x56x56
        self.decoder_4 = nn.Sequential()
        self.unpool.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_4.add_module('conv_2_1', conv_bn_relu(256, 128, kernel=3, stride=1, padding=1))
        self.decoder_4.add_module('conv_2_2',conv_bn_relu(128, 128, kernel=3, stride=1, padding=1))
        self.decoders.append(self.decoder_4)
        # output 128x112x112

        # input 128x112x112
        self.decoder_5 = nn.Sequential()
        self.unpool.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_5.add_module('conv_1_1', conv_bn_relu(128, 64, kernel=3, stride=1, padding=1))
        self.decoder_5.add_module('conv_1_2',conv_bn_relu(64, self.num_classes, kernel=3, stride=1, padding=1))
        self.decoders.append(self.decoder_5)
        # output num_classesx224x224

        #self.fc_classifier = nn.Linear(224*224, 100)    

         
    def forward(self, input):
        # encoder
        num = 0
        x = input
        n, c, h, w = x.shape
        for num in range(5):
            x = self.encoders[num](x)
            x, self.pool_idx[num] = self.pool[num](x)
        for num in range(5):
            x = self.unpool[num](x, self.pool_idx[4 - num])
            x = self.decoders[num](x)
        x = x.view((x.size()[0], -1))   
        #output = self.fc_classifier(x)
        output = x
        output = output.view((n, self.num_classes, h, w))
        return output

