# import torch
# from torch import nn
# from torchvision.models.vgg import vgg16
#
#
# class GeneratorLoss(nn.Module):
#     def __init__(self):
#         super(GeneratorLoss, self).__init__()
#         vgg = vgg16(pretrained=True)
#         loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
#         for param in loss_network.parameters():
#             param.requires_grad = False
#         self.loss_network = loss_network
#         #self.mse_loss = nn.MSELoss()
#         self.L1_loss = nn.L1Loss()
#         # self.tv_loss = TVLoss()
#
#     def forward(self, out_labels, out_images, target_images):
#