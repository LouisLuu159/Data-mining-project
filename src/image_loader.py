from multiprocessing import freeze_support

import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision.transforms import transforms
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
import PIL
from pathlib import Path


class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):

        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        return (img, label, path)


num_classes = 5
batch_size = 100
num_of_workers = 5

DATA_PATH_TRAIN = Path(
    'D:\\Data_mining\\Dataset\\test_load_image')
DATA_PATH_TEST = Path('D:\\Data_mining\\Dataset\\test')

trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(28),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ImageFolderWithPaths(root=DATA_PATH_TRAIN, transform=trans)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_of_workers)


def imshow(img, paths):
    img = img / 2 + 0.5  # unnormalize
    #npimg = img.numpy()
    original_img = plt.imread(paths[0])

    fig = plt.figure()
    img_no_trans = fig.add_subplot(1, 2, 1)
    img_no_trans.imshow(original_img)
    img_no_trans.set_title("Img before transform")

    img_trans = fig.add_subplot(1, 2, 2)
    img_trans.imshow(np.transpose(img[0].numpy(), (1, 2, 0)))
    img_trans.set_title("Img after transform")

    plt.show()


def main():
    # get some random training images
    dataiter = iter(train_loader)
    images, labels, paths = dataiter.next()
    print(paths[0])

    # show images
    imshow(images, paths)


if __name__ == "__main__":
    main()
