import torch.nn as nn

BATCHNORM = True # Can be used to mesure performance difference
K_CONV = 3 # Kernel size for convolutional layer
P_CONV = 1 # Padding size for convolutional layer
K_POOL = 2 # Kernel size for max polling layer
S_POOL = 2 # Stride size for max polling layer


def encoder_layer(in_c, out_c, depth = 2):
    layer = []
    for i in range(depth):
        if i == 0:
            layer.append(nn.Conv2d(in_c, out_c, kernel_size=K_CONV, padding=P_CONV))
        else:
            layer.append(nn.Conv2d(out_c, out_c, kernel_size=K_CONV, padding=P_CONV))
        if BATCHNORM:
            layer.append(nn.BatchNorm2d(out_c))
        layer.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layer)


def maxpool_layer():
    layer = nn.MaxPool2d(kernel_size=K_POOL, stride=S_POOL, return_indices=True)
    return layer
            

def unpool_layer():
    layer = nn.MaxUnpool2d(kernel_size=K_POOL, stride=S_POOL)
    return layer


def decoder_layer(in_c, out_c, depth=2, not_final=True):
    layer = []
    for i in range(depth):
        if i == depth - 1:
            layer.append(nn.Conv2d(in_c, out_c, kernel_size=K_CONV, padding=P_CONV))
            if not_final:
                if BATCHNORM:
                    layer.append(nn.BatchNorm2d(out_c))
                layer.append(nn.ReLU(inplace=True))
        else:
            layer.append(nn.Conv2d(in_c, in_c, kernel_size=K_CONV, padding=P_CONV))
            if BATCHNORM:
                layer.append(nn.BatchNorm2d(in_c))
            layer.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layer)


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv0 = encoder_layer(3, 64)
        self.pool0 = maxpool_layer()
        self.enc_conv1 = encoder_layer(64, 128)
        self.pool1 = maxpool_layer()
        self.enc_conv2 = encoder_layer(128, 256, depth=3)
        self.pool2 = maxpool_layer()
        self.enc_conv3 = encoder_layer(256, 512, depth=3)
        self.pool3 = maxpool_layer()

        # Bottleneck
        self.bottlenecK_CONV = encoder_layer(512, 512, depth=3) 
        self.bottlenecK_POOL = maxpool_layer()
        self.bottleneck_unpool = unpool_layer()
        self.bottleneck_dec_conv = decoder_layer(512, 512, depth=3) 

        # Decoder
        self.upsample0 = unpool_layer()
        self.dec_conv0 = decoder_layer(512, 256, depth=3)
        self.upsample1 = unpool_layer()
        self.dec_conv1 = decoder_layer(256, 128, depth=3)
        self.upsample2 = unpool_layer()
        self.dec_conv2 = decoder_layer(128, 64)
        self.upsample3 = unpool_layer()
        self.dec_conv3 = decoder_layer(64, 1, not_final=False) # No activation

    def forward(self, x):
        # encoder
        e0, ind0 = self.pool0(self.enc_conv0(x))
        e1, ind1 = self.pool1(self.enc_conv1(e0))#ind - indices
        e2, ind2 = self.pool2(self.enc_conv2(e1))
        e3, ind3 = self.pool3(self.enc_conv3(e2))

        # bottleneck
        b0 = self.bottlenecK_CONV(e3)
        b1, indb = self.bottlenecK_POOL(b0)
        b2 = self.bottleneck_unpool(b1, indb)
        b3 = self.bottleneck_dec_conv(b2)

        # decoder
        d0 = self.dec_conv0(self.upsample0(b3, ind3))
        d1 = self.dec_conv1(self.upsample1(d0, ind2))
        d2 = self.dec_conv2(self.upsample2(d1, ind1))
        output = self.dec_conv3(self.upsample3(d2, ind0))  
        return output 
