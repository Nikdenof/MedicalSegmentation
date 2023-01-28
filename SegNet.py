import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_PADDING = 1
CONV_STRIDE = 1
CONV_KERNEL= 3
POOL_STRIDE = 2
POOL_KERNEL = 2

INPUT_SIZE = 3
OUTPUT_SIZE = 1 # Class output (0 - nothing, 1 - segmented something)

BATCHNORM = False # Can be used to mesure performance difference

def encoder_layer_s(in_c, out_c):
    k_conv = 3 # Kernel size for convolutional layer
    p_conv = 1 # Padding size for convolutional layer
    if BATCHNORM:
        layer = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=k_conv, padding=p_conv),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=k_conv, padding=p_conv),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                )
    else:
        layer = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=k_conv, padding=p_conv),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=k_conv, padding=p_conv),
                    nn.ReLU(inplace=True)
                )
    return layer

def encoder_layer_l(in_c, out_c):
    k_conv = 3 # Kernel size for convolutional layer
    p_conv = 1 # Padding size for convolutional layer
    if BATCHNORM:
        layer = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=k_conv, padding=p_conv),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=k_conv, padding=p_conv),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                    nn.Conv2d(out_c, out_c, kernel_size=k_conv, padding=p_conv),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                )
    else:
        layer = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=k_conv, padding=p_conv),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=k_conv, padding=p_conv),
                    nn.ReLU(inplace=True)
                    nn.Conv2d(out_c, out_c, kernel_size=k_conv, padding=p_conv),
                    nn.ReLU(inplace=True)
                )
    return layer

def maxpool_layer():
    k_pool = 2 # Kernel size for max polling layer
    s_pool = 2 # Stride size for max polling layer
    layer = nn.MaxPool2d(kernel_size=k_pool, stride=s_pool, return_indices=True)
    
    return layer
            

def bottleneck():
    """
    SOMETHING WOOOOOOOOOOOOOOOOOOOOOOOOOOOO

    """
    c = 512 # The number of in and out channels in bottleneck is the same
    k_conv = 3 # Kernel size for convolutional layer
    p_conv = 1 # Padding size for convolutional layer
    if BATCHNORM:
        layer = nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=k_conv, padding=p_conv),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, kernel_size=k_conv, padding=p_conv),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, kernel_size=k_conv, padding=p_conv),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True)
                )
    else:
        layer = nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=k_conv, padding=p_conv),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, kernel_size=k_conv, padding=p_conv),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, kernel_size=k_conv, padding=p_conv),
                    nn.ReLU(inplace=True)
                )
    return layer


def unpool_layer():
    k_pool = 2 # Kernel size for max polling layer
    s_pool = 2 # Stride size for max polling layer
    layer = nn.MaxUnpool2d(kernel_size=k_pool, stride=s_pool)
    
    return layer


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc_conv0 = encoder_layer_s(3, 64)
        self.pool0 = maxpool_layer()
        self.enc_conv1 = encoder_layer_s(64, 128)
        self.pool1 = maxpool_layer()
        self.enc_conv2 = encoder_layer_l(128, 256)
        self.pool2 = maxpool_layer()
        self.enc_conv3 = encoder_layer_l(256, 512)
        self.pool3 = maxpool_layer()
        # Bottleneck
        self.bottleneck_conv = bottleneck()
        self.bottleneck_pool = maxpool_layer()
        self.bottleneck_unpool = unpool_layer()
        self.bottleneck_dec_conv = bottleneck()

        self.upsample = unpool_layer()

    def forward(self, x):
        # encoder
        e0, ind0 = self.pool0(self.enc_conv0(x))
        e1, ind1 = self.pool1(self.enc_conv1(e0))#ind - indices
        e2, ind2 = self.pool2(self.enc_conv2(e1))
        e3, ind3 = self.pool3(self.enc_conv3(e2))

        # bottleneck
        b0 = self.bottleneck_conv(e3)
        b1, indb = self.bottleneck_pool(b0)
        b2 = self.bottleneck_unpool(b1, indb)
        b3 = self.bottleneck_dec_conv(b2)

        # decoder
        d0 = self.dec_conv0(self.upsample0(b3, ind3))
        d1 = self.dec_conv1(self.upsample1(d0, ind2))
        d2 = self.dec_conv2(self.upsample2(d1, ind1))
        d3 = self.dec_conv3(self.upsample3(d2, ind0))  # no activation
        #the output image is going to have the same number of dimensions as the input image
        return d3
