import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# 01. Origin UNET Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):    # Conv + BatchNorm + ReLU
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

    # Contracting Path    
        # Encoder 1
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Encoder 2
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Encoder 3
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Encoder 4
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Encoder 5
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
        
    # Expanding Path
        # Decoder 5
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)
        
        # Decoder 4
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512) #  skip connection 으로 *2
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        # Decoder 3
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256) # skip connection 으로 *2
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        
        # Decoder 2
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128) # skip connection 으로 *2
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)
        
        # Decoder 1
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64) # skip connection 으로 *2
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # number of class
        self.fc = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        # Encoder - Contracting Path
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        # Decoder - Expanding Path
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)      # dim=0 > batch, dim=1 > channel, dim=2 > height, dim3 > width
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        output = self.fc(dec1_1)
        # output = torch.sigmoid(output)  # activation function
        # output = self.softmax(output)

        return output


# 02. Attention UNET Model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        output = self.conv(x)

        return output
 
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        output = self.up(x)
        return output

class AttentionBlock(nn.Module):
    """
        Attention block with learnable parameters.
    """

    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.attn_score = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)

        attn_score = self.relu(g1 + x1)
        attn_score = self.attn_score(attn_score)
        out = skip_connection*attn_score

        return out
    
class AttentionUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=32):
        super(AttentionUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.Upconv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder

        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        # Decoder
        d5 = self.Up5(e5)

        s4 = self.Att5(gate=d5, skip_connection=e4) # skip connection
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.Upconv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        output = self.Conv(d2)
        output = self.softmax(output)
        # output = self.sigmoid(output)

        return output





# 03. DUCK-NET (https://arxiv.org/pdf/2311.02239.pdf)

# Convoultion Block
class conv_block_2D(nn.Module):
    def __init__(self, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same', half_channel=False):
        super(conv_block_2D, self).__init__()

        self.conv_block = nn.ModuleList()
        self.half_channel = half_channel
        self.filters = filters
        
        for i in range(0, repeat):
            if i == 1:
                if self.half_channel:
                    self.filters = int(self.filters/2)
                    
                self.half_channel = False

            if block_type == 'separated':
                self.conv_block.append(
                    separated_conv2D_block(self.filters, size=size, padding=padding, half_channel=self.half_channel)
                )
            elif block_type == 'duckv2':
                self.conv_block.append(
                    duckv2_conv2D_block(self.filters, size=size, half_channel=self.half_channel)
                    )
                
            elif block_type == 'midscope':
                self.conv_block.append(
                    midscope_conv2D_block(self.filters, half_channel=self.half_channel)
                )
            elif block_type == 'widescope':
                self.conv_block.append(
                    widescope_conv2D_block(self.filters, half_channel=self.half_channel)
                )           
            elif block_type == 'resnet':
                self.conv_block.append(
                    resnet_conv2D_block(self.filters, self.half_channel, dilation_rate)
                )
            elif block_type == 'conv':
                self.conv_block.append(
                    nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=size, padding=padding)
                )
                self.conv_block.append(nn.ReLU())
            elif block_type == 'double_convolution':
                self.conv_block.append(
                    double_convolution_with_batch_normalization(self.filters, dilation_rate)
                )
            else:
                print('HERE')
                return None
    def forward(self, x):
        for i in range(len(self.conv_block)):
            x = self.conv_block[i](x)
        
        return x

class separated_conv2D_block(nn.Module):
    def __init__(self, filters, half_channel, size=3, padding='same'):
        super(separated_conv2D_block, self).__init__()

        if half_channel:
            self.out_channel = int(filters/2)
        else:
            self.out_channel = filters
        
        self.conv_block = nn.Sequential( 
            nn.Conv2d(in_channels=filters, out_channels=self.out_channel, kernel_size=(1, size), padding=padding),
            nn.BatchNorm2d(self.out_channel), 
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=(size,1), padding=padding),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        output = self.conv_block(x)
        return output

# DUCK Block
class duckv2_conv2D_block(nn.Module):
    def __init__(self, filters, half_channel, size, padding='same'):
        super(duckv2_conv2D_block, self).__init__()
        self.filters = filters

        if half_channel:
            self.filters = int(filters/2)

        self.x_0 = nn.BatchNorm2d(filters)
        self.x_1 = widescope_conv2D_block(filters, half_channel=half_channel)
        self.x_2 = midscope_conv2D_block(filters, half_channel=half_channel)
        self.x_3 = conv_block_2D(filters, 'resnet', repeat=1, half_channel=half_channel)
        self.x_4 = conv_block_2D(filters, 'resnet', repeat=2, half_channel=half_channel)
        self.x_5 = conv_block_2D(filters, 'resnet', repeat=3, half_channel=half_channel)
        self.x_6 = separated_conv2D_block(filters, size=6, padding='same', half_channel=half_channel)
        self.x_7 = nn.BatchNorm2d(self.filters)

    def forward(self, x):
        x = self.x_0(x)     # BN
        x1 = self.x_1(x)
        x2 = self.x_2(x)
        x3 = self.x_3(x)
        x4 = self.x_4(x)
        x5 = self.x_5(x)
        x6 = self.x_6(x)

        x = x1 + x2 + x3 + x4 + x5 + x6
        output = self.x_7(x) 
        
        return output

# Mid Scope Block
class midscope_conv2D_block(nn.Module):
    def __init__(self, filters, half_channel):
        super(midscope_conv2D_block, self).__init__()
        
        if half_channel:
            self.out_channel = int(filters/2)
        else:
            self.out_channel = filters

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=self.out_channel, kernel_size=3, padding='same', dilation=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding='same', dilation=2),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU()
        )
    def forward(self, x):
        output = self.conv_block(x)

        return output

# Wide Scope Block
class widescope_conv2D_block(nn.Module):
    def __init__(self, filters, half_channel):
        super(widescope_conv2D_block, self).__init__()
        
        if half_channel:
            self.out_channel = int(filters/2)
        else:
            self.out_channel = filters

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=self.out_channel, kernel_size=3, padding='same', dilation=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding='same', dilation=2),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding='same', dilation=3),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU()            
        )
    def forward(self, x):
        output = self.conv_block(x)

        return output

# Residual Block
class resnet_conv2D_block(nn.Module):
    def __init__(self, filters, half_channel, dilation_rate):
        super(resnet_conv2D_block, self).__init__()

        if half_channel:
            self.out_channel = int(filters/2)
        else:
            self.out_channel = filters
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=self.out_channel, kernel_size=1, padding='same', dilation=dilation_rate),
            nn.ReLU()
        )   # residual layer

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=self.out_channel, kernel_size=1, padding='same', dilation=dilation_rate),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding='same', dilation=dilation_rate),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU()
        )   # convolution layer

        self.conv_final = nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        x1 = self.conv_block1(x)
        
        x = self.conv_block2(x)
        output = x + x1
        output = self.conv_final(output)

        return output



class double_convolution_with_batch_normalization(nn.Module):
    def __init__(self, filters, half_channel, dilation_rate):
        super(double_convolution_with_batch_normalization, self).__init__()

        if half_channel:
            self.out_channel = int(filters/2)
        else:
            self.out_channel = filters

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=self.out_channel, kernel_size=3, padding='same', dilation=dilation_rate),
            nn.Batchnorm2d(self.out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding='same', dilation=dilation_rate),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.conv_block(x)

        return output



class DuckNet(nn.Module):
    def __init__(self, starting_filters):
        super(DuckNet, self).__init__()

        # Down Sampling
        self.change_channels = nn.Conv2d(in_channels=1, out_channels=starting_filters, kernel_size=1, stride=1)
        self.downsample_1 = nn.Conv2d(in_channels=starting_filters, out_channels=starting_filters*2, kernel_size=2, stride=2)
        self.downsample_2 = nn.Conv2d(in_channels=starting_filters*2, out_channels=starting_filters*4, kernel_size=2, stride=2)
        self.downsample_3 = nn.Conv2d(in_channels=starting_filters*4, out_channels=starting_filters*8, kernel_size=2, stride=2)
        self.downsample_4 = nn.Conv2d(in_channels=starting_filters*8, out_channels=starting_filters*16, kernel_size=2, stride=2)
        self.downsample_5 = nn.Conv2d(in_channels=starting_filters*16, out_channels=starting_filters*32, kernel_size=2, stride=2)
        
        # Down Duck Block
        self.downduck_0 = conv_block_2D(starting_filters, block_type='duckv2')
        self.downduck_1 = conv_block_2D(starting_filters*2, block_type='duckv2')
        self.downduck_2 = conv_block_2D(starting_filters*4, block_type='duckv2')
        self.downduck_3 = conv_block_2D(starting_filters*8, block_type='duckv2')
        self.downduck_4 = conv_block_2D(starting_filters*16, block_type='duckv2')

        # Down Sampling (Duck block)
        self.downsample_d_1 = nn.Conv2d(in_channels=starting_filters, out_channels=starting_filters*2, kernel_size=2, stride=2)
        self.downsample_d_2 = nn.Conv2d(in_channels=starting_filters*2, out_channels=starting_filters*4, kernel_size=2, stride=2)
        self.downsample_d_3 = nn.Conv2d(in_channels=starting_filters*4, out_channels=starting_filters*8, kernel_size=2, stride=2)
        self.downsample_d_4 = nn.Conv2d(in_channels=starting_filters*8, out_channels=starting_filters*16, kernel_size=2, stride=2)
        self.downsample_d_5 = nn.Conv2d(in_channels=starting_filters*16, out_channels=starting_filters*32, kernel_size=2, stride=2)
        
        # Res Block
        self.res_1 = conv_block_2D(starting_filters*32, block_type='resnet', repeat=2)
        self.res_2 = conv_block_2D(starting_filters*32, block_type='resnet', repeat=2, half_channel=True)

        # Up Duck Block
        self.upduck_0 = conv_block_2D(starting_filters, 'duckv2', repeat=1)
        self.upduck_1 = conv_block_2D(starting_filters*2, 'duckv2', repeat=1, half_channel=True)
        self.upduck_2 = conv_block_2D(starting_filters*4, 'duckv2', repeat=1, half_channel=True)
        self.upduck_3 = conv_block_2D(starting_filters*8, 'duckv2', repeat=1, half_channel=True)
        self.upduck_4 = conv_block_2D(starting_filters*16, 'duckv2', repeat=1, half_channel=True)

        self.output = nn.Conv2d(in_channels=starting_filters, out_channels=32, kernel_size=1, stride=1, padding='same')
        self.softmax = nn.Softmax2d()

    def forward(self, x):                   # Input shape: (1, 1, 256, 256) ==> (Batch, Channels, Height, Weight)
        x = self.change_channels(x)

    # Down sampling (Origin Image)
        down_1 = self.downsample_1(x)
        down_2 = self.downsample_2(down_1)
        down_3 = self.downsample_3(down_2)
        down_4 = self.downsample_4(down_3)
        down_5 = self.downsample_5(down_4)

    # Down sampling (Origin Image + Duck Block)
        duck_0 = self.downduck_0(x)
        # duck block down 1 (blue arrow in article)
        down_d_1 = self.downsample_d_1(duck_0)
        downadd_1 = down_1 + down_d_1  # down + duck down summation
        downduck_1 = self.downduck_1(downadd_1)

        # duck block down 2
        down_d_2 = self.downsample_d_2(downduck_1)
        downadd_2 = down_2 + down_d_2
        downduck_2 = self.downduck_2(downadd_2)

        # duck block down 3
        down_d_3 = self.downsample_d_3(downduck_2)
        downadd_3 = down_3 + down_d_3
        downduck_3 = self.downduck_3(downadd_3)

        # duck block down 4
        down_d_4 = self.downsample_d_4(downduck_3)
        downadd_4 = down_4 + down_d_4
        downduck_4 = self.downduck_4(downadd_4)

        # duck block down 5
        down_d_5 = self.downsample_d_5(downduck_4)
        downadd_5 = down_5 + down_d_5

        # 2 res blocks (yellow arrow in article)
        res_1 = self.res_1(downadd_5)
        res_2 = self.res_2(res_1)

    # Up sampling (Origin Image + Duck Block / Neareset)
        up_4 = nn.Upsample(scale_factor=2, mode='nearest')(res_2)
        upadd_4 = downduck_4 + up_4
        upduck_4 = self.upduck_4(upadd_4)

        up_3 = nn.Upsample(scale_factor=2, mode='nearest')(upduck_4)
        upadd_3 = downduck_3 + up_3
        upduck_3 = self.upduck_3(upadd_3)

        up_2 = nn.Upsample(scale_factor=2, mode='nearest')(upduck_3)
        upadd_2 = downduck_2 + up_2
        upduck_2 = self.upduck_2(upadd_2)

        up_1 = nn.Upsample(scale_factor=2, mode='nearest')(upduck_2)
        upadd_1 = downduck_1 + up_1
        upduck_1 = self.upduck_1(upadd_1)

        up_0 = nn.Upsample(scale_factor=2, mode='nearest')(upduck_1)
        upadd_0 = duck_0 + up_0
        upduck_0 = self.upduck_0(upadd_0)

        output = self.output(upduck_0)
        output = self.softmax(output)
        return output
    
