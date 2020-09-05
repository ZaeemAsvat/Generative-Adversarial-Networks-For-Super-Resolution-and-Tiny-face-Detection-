from torch.nn import PixelShuffle, ReLU, Sigmoid, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, AdaptiveAvgPool2d
import math
import torch

def commonConv(in_channels, out_channels, kernel_size, bias=True):
    return Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class RCAN_Args():

    def __init__(self):
        self.num_features = -1
        self.num_res_groups = -1
        self.num_res_blocks = -1
        self.kernel_size = -1
        self.num_colours = -1
        self.reduction = -1
        self.scale = -1

class MeanShift(Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class Upsampler(Sequential):

    """ Up-sampling layer (Sub-pixel) """

    def __init__(self, conv, scale, numFeatures, kernelSize, bias=True):

        moduleList = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for i in range(int(math.log(scale, 2))):
                moduleList.append(conv(numFeatures, 4 * numFeatures, kernelSize, bias))
                moduleList.append(PixelShuffle(2))
                # if act: m.append(act())
        elif scale == 3:
            moduleList.append(conv(numFeatures, 9 * numFeatures, kernelSize, bias))
            moduleList.append(PixelShuffle(3))
            # if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*moduleList)


class CALayer(Module):

    """ Channel Attention (CA) Layer """

    def __init__(self, channel, reduction=16):

        super(CALayer, self).__init__()

        # global average pooling: feature --> point
        self.avg_pool = AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = Sequential(
                Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                ReLU(inplace=True),
                Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                Sigmoid()
        )

    def forward(self, x):
        return x * self.conv_du(self.avg_pool(x))


class RCAB(Module):

    """ Residual Channel Attention Block (RCAB) """

    def __init__(self, conv, numFeatures, kernelSize, reduction, bias=True, resScale=1):

        super(RCAB, self).__init__()

        self.body = Sequential(
            conv(numFeatures, numFeatures, kernelSize, bias=bias),
            ReLU(True),
            conv(numFeatures, numFeatures, kernelSize, bias=bias),
            CALayer(numFeatures, reduction))
        self.res_scale = resScale

    def forward(self, x):
        residual = self.body(x)
        return x + residual


class ResidualGroup(Module):

    """ Residual Group (RG) """

    def __init__(self, conv, numFeatures, kernelSize, reduction, numResBlocks):

        super(ResidualGroup, self).__init__()

        modulesList = []
        for i in range(numResBlocks):
            modulesList.append(RCAB(conv, numFeatures, kernelSize, reduction, bias=True, resScale=1))
        modulesList.append(conv(numFeatures, numFeatures, kernelSize))
        self.body = Sequential(*modulesList)

    def forward(self, x):
        residual = self.body(x)
        return x + residual

class Generator(Module):

    def __init__(self, args):

        super(Generator, self).__init__()

        # args = RCAN_Args()

        numResGroups = args.num_res_groups
        numResBlocks = args.num_res_blocks
        numFeatures = args.num_features
        kernelSize = args.kernel_size

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)

        # define shallow feature extraction module
        self.shallowFeatureExtract = Sequential(commonConv(args.num_colours, numFeatures, kernelSize))

        # define deep feature extraction module
        modulesList = []
        for i in range(numResGroups):
            modulesList.append(ResidualGroup(conv=commonConv, numFeatures=numFeatures, kernelSize=kernelSize, reduction=args.reduction, numResBlocks=numResBlocks))
        modulesList.append(commonConv(numFeatures, numFeatures, kernelSize))
        self.deepFeatureExtract = Sequential(*modulesList)

        # define upsample module
        self.upsampler = Sequential(Upsampler(commonConv, scale=args.scale, numFeatures=numFeatures, kernelSize=kernelSize, bias=True))

        # define reconstruction module
        self.reconstruction = Sequential(commonConv(numFeatures, args.num_colours, kernelSize))

        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

    def forward(self, x):

        x = self.sub_mean(x)

        # extract shallow features
        shallowFeatures = self.shallowFeatureExtract(x)

        # extract deep features
        deepFeatures = self.deepFeatureExtract(shallowFeatures)
        deepFeatures += shallowFeatures

        # up-sample and reconstruction
        result = self.reconstruction(self.upsampler(deepFeatures))

        # x = self.add_mean(x)

        # return final reconstructed features
        return self.add_mean(result)
