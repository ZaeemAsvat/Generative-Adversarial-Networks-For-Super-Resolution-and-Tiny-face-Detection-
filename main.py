from generator import Generator, RCAN_Args
import torchvision
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from os import listdir
from os.path import join
from tqdm import tqdm
import numpy as np
import math
from torchvision.utils import save_image

# Hyper-parameters
EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
TRAIN_DATA_PATH = 'DIV2K/train/'
TEST_DATA_PATH = 'DIV2K/DIV2K_test_LR_bicubic/'
TRANSFORM_IMG = torchvision.transforms.Compose([
torchvision.transforms.ToTensor()])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def psnr(hr, sr, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    hr = hr.cpu().detach().numpy()
    sr = sr.cpu().detach().numpy()
    img_diff = sr - hr
    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


# Load data
CROP_SIZE = 88
UPSCALE_FACTOR = 4

train_data = TrainDatasetFromFolder('DIV2K/train/HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_data = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR)
val_data_loader = data.DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

crop_size = calculate_valid_crop_size(crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
hr_transform = train_hr_transform(crop_size=crop_size)
lr_transform = train_lr_transform(crop_size=crop_size, upscale_factor=UPSCALE_FACTOR)



if __name__ == '__main__':

    # ----------------------- Training -----------------------------
    rcan_args = RCAN_Args()
    rcan_args.num_res_groups = 10
    rcan_args.num_res_blocks = 20
    rcan_args.num_features = 64
    rcan_args.scale = 4
    rcan_args.kernel_size = 3
    rcan_args.num_colours = 3 # (i think)
    rcan_args.reduction = 16
    rcan_generator = Generator(rcan_args)

    optimizer = torch.optim.Adam(params=rcan_generator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=0.00000001)
    loss_func = nn.MSELoss() # default


    def train(model, train_data_loader):

        running_loss = 0.0
        running_psnr = 0.0
        model.train()
        train_bar = tqdm(train_data_loader)
        for lr, hr in train_bar:

            hr_image = Variable(hr)
            lr_image = Variable(lr)

            optimizer.zero_grad()
            sr_image = model(lr_image)
            loss = loss_func(sr_image, hr_image)

            # back-prop and update params
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_psnr += psnr(hr_image, sr_image)

        final_loss = running_loss / len(train_data_loader.dataset)
        final_psnr = running_psnr / int(len(train_data) / train_data_loader.batch_size)

        print('final loss', final_loss)
        print('final psnr', final_psnr)

        # is this actually gonma work? Idk
        return final_loss, final_psnr


    def validate(model, valid_data_loader, epoch):

        running_loss = 0.0
        running_psnr = 0.0
        model.eval()
        val_bar = tqdm(valid_data_loader)
        with torch.no_grad():
            for lr, hr in val_bar:

                hr_image = Variable(hr)
                lr_image = Variable(lr)

                optimizer.zero_grad()
                sr_image = model(lr_image)
                loss = loss_func(sr_image, hr_image)

                # back-prop and update params
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_psnr += psnr(hr_image, sr_image)

                sr_image = sr_image.cpu()
                save_image(sr_image, f"../outputs/val_sr{epoch}.png")

            final_loss = running_loss / len(train_data_loader.dataset)
            final_psnr = running_psnr / int(len(train_data) / train_data_loader.batch_size)

        print('final loss', final_loss)
        print('final psnr', final_psnr)

        # is this actually gonma work? Idk
        return final_loss, final_psnr
