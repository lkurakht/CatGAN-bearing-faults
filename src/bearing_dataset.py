import numpy as np
import torch

from scipy.io import loadmat
from scipy.signal import stft

from scipy import ndimage, misc


def Normalize(arr: np.array) -> np.array:
    """
    Min-max normalization of array to [0,1]
    :param arr: array to normalize
    :return:
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def Normalize(arr: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalization of tensor to [0,1]
    :param arr: array to normalize
    :return:
    """
    arr -= arr.min()
    arr /= arr.max()
    return arr

def ScaleToSquare(arr: np.ndarray, size: int) -> np.ndarray:
    """
    Rescale 2D numpy array as image to square with specified size
    :param arr: 2D numpy array (image)
    :param size: width and height of a new rescaled image
    :return: 2D numpy array containing rescaled image
    """
    return ndimage.zoom(arr, (size / arr.shape[0], size / arr.shape[1]))


class BearingDataset(torch.utils.data.Dataset):
    def __init__(self, transform = None, target_transform = None, img_size = 64, segment_size = 1024, step = 128,
                 Nw = 256, No = 250,
                 fs = 12000., type = 'A'):
        """
        Initialize bearing dataset from https://engineering.case.edu/bearingdatacenter/download-data-file
        :param img_size: output time-frequency map image size after rescaling
        :param segment_size: number of time samples to process one time-frequency map image
        :param step: number of shift between neighbour segments
        :param Nw: window function width
        :param No: neighbour windows overlap
        :param fs: sample frequency, Hz
        :param type: Dataset type according to https://sci-hub.ru/https://www.sciencedirect.com/science/article/abs/pii/S0016003220302544
        Default dataset type is 'A', which have 4 classes: normal data and 3 types of faults
        TODO: make an option to download datafiles directly from Bearing Data Center
        """
        self.img_size = img_size
        self.segment_size = segment_size
        self.step = step
        self.Nw = Nw  # window width from 4.3 section
        self.No = No  # window overlap from 4.3 section
        self.fs = fs  # 12KHz
        self.type = type  # Let use only A, B, C datasets

        self.normal_label = 'Noprmal'
        self.normal_filename = '../datasets/A/None/97.mat'
        self.a_labels = [self.normal_label, 'BA007', 'BA014', 'BA021']
        self.a_filenames = [self.normal_filename, '../datasets/A/BA/0.007/118.mat',
                            '../datasets/A/BA/0.014/185.mat',
                            '../datasets/A/BA/0.021/222.mat']
        self.b_labels = [self.normal_label, 'BA007', 'IR007', 'OR007']
        self.b_filenames = [self.normal_filename, '../datasets/A/BA/0.007/118.mat',
                            '../datasets/C/IR/0.007/105.mat',
                            '../datasets/C/OR/0.007/130.mat']
        self.c_labels = [self.normal_label, 'BA007', 'BA014', 'BA021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014',
                         'OR021']
        self.c_filenames = [self.normal_filename, '../datasets/A/BA/0.007/118.mat',
                            '../datasets/A/BA/0.014/185.mat',
                            '../datasets/A/BA/0.021/222.mat', '../datasets/C/IR/0.007/105.mat',
                            '../datasets/C/IR/0.014/169.mat',
                            '../datasets/C/IR/0.021/209.mat', '../datasets/C/OR/0.007/130.mat',
                            '../datasets/C/OR/0.014/197.mat',
                            '../datasets/C/OR/0.021/234.mat']
        if self.type == 'A':
            self.labels = self.a_labels
            self.filenames = self.a_filenames
        elif self.type == 'B':
            self.labels = self.b_labels
            self.filenames = self.b_filenames
        elif self.type == 'C':
            self.labels = self.c_labels
            self.filenames = self.c_filenames
        else:
            raise ValueError('Unsupported dataset type.')

        if len(self.labels) != len(self.filenames):
            raise RuntimeError('labels and filenames number doesnt matchs')

        self.img_labels = []

        ### "amount of samples per class is expected to be the same for all k categories", cropping to the minimum length
        minimum_length = 1e10
        for idx, filename in enumerate(self.filenames):
            signal = loadmat(filename)
            drive_end_key = [k for k in signal.keys() if 'DE_time' in k]
            if not len(drive_end_key):
                continue
            minimum_length = min(minimum_length, len(signal[drive_end_key[0]].ravel()))

        for idx, filename in enumerate(self.filenames):
            signal = loadmat(filename)
            drive_end_key = [k for k in signal.keys() if 'DE_time' in k]
            if not len(drive_end_key):
                continue
            de_signal = signal[drive_end_key[0]].ravel()
            for i in range(0, (minimum_length - self.segment_size), self.step):
                pack_data = de_signal[i: i + self.segment_size]
                freq, time, zmag = stft(pack_data, fs, window = 'hann', nperseg = self.Nw, noverlap = self.No)
                scaled_zmag = ScaleToSquare(np.abs(zmag), self.img_size)
                normalized_zmag = Normalize(scaled_zmag)
                self.img_labels.append((normalized_zmag, idx))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.img_labels[idx]

    def get_type(self) -> str:
        """
        Get type of dataset
        :return: Dataset type ('A', 'B' or 'C')
        """
        return self.type

    def get_cats_num(self) -> int:
        """
        :return: Number of classifying categories
        """
        return len(self.labels)

    def getlabel(self, idx) -> int:
        """
        Get label for image with specified index
        :param idx: image index
        :return: image label
        """
        return self.labels[idx]

    def getimgs(self, labelidx):
        """
        Get list of all images with specified label
        :param labelidx: label (0 for normal data)
        :return: list of all images with specified label
        """
        return [image for (image, label) in self.img_labels if label == labelidx]
