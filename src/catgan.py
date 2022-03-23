import torch
from torch import nn
from bearing_dataset import Normalize

class Classifier(nn.Module):
    def __init__(self, cat_num = 4):
        """
        Initialize classifier
        :param cat_num: Number of classes to classify
        """
        super().__init__()
        self.cat_num = cat_num
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=4, padding=2)
        self.norm1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.25)
        self.act = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.norm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.norm3 = nn.BatchNorm2d(128)

        self.fc1 = torch.nn.Linear(128 * 4 * 4, self.cat_num)

    def forward(self, x):
        #        print('input: ', x.shape)
        x = self.dropout(self.norm1(self.act(self.conv1(x))))
        #        print('after first layer: ', x.shape)

        x = self.dropout(self.norm2(self.act(self.conv2(x))))
        #        print('after second layer: ', x.shape)

        x = self.dropout(self.norm3(self.act(self.conv3(x))))
        #        print('after third layer: ', x.shape)

        x = x.view(x.shape[0], -1)
        #        print('after flattening: ', x.shape)

        x = self.fc1(x)
        #        print('after classifying: ', x.shape)

        return x

    def get_latent_features(self, x):
        #        print('input: ', x.shape)
        x = self.dropout(self.norm1(self.act(self.conv1(x))))
        #        print('after first layer: ', x.shape)

        x = self.dropout(self.norm2(self.act(self.conv2(x))))
        #        print('after second layer: ', x.shape)

        x = self.dropout(self.norm3(self.act(self.conv3(x))))
        #        print('after third layer: ', x.shape)

        x = x.view(x.shape[0], -1)
        #        print('after flattening: ', x.shape)
        return x

class Generator(nn.Module):
    def __init__(self):
        """
        Initialize generator
        """
        super().__init__()

        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(200, 200, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(200)
        )

        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(200, 64, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 2),
            nn.Tanh()
        )

        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size = 4, stride = 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        # for idx in range(x.shape[0]):
        #     x[idx] = Normalize(x[idx])
        return x
