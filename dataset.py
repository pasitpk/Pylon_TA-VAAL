import cv2
import torch
from albumentations import *
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

cv2.setNumThreads(1)

# chestxray's
MEAN = [0.4984]
SD = [0.2483]

def get_array_loader(X, y, batch_size, transform, shuffle=False):
    return DataLoader(CXRArray(X, y, transform), batch_size=batch_size, shuffle=shuffle, num_workers=2)


class CXRArray(Dataset):
    def __init__(self, X, y, transform=None):

        self.X = X
        self.y = y
        self.transform = transform
        self.support = len(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        
        img = self.X[i]

        if self.transform:
            _res = self.transform(image=img)
            img = _res['image']

        labels = torch.from_numpy(self.y[i]).float()

        return img, labels


def make_transform(
        augment,
        size=256,
        rotate=90,
        p_rotate=0.5,
        brightness=0.5,
        contrast=0.5,
        min_size=0.7,
        interpolation='cubic',
):
    inter_opts = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
    }
    inter = inter_opts[interpolation]

    trans = []
    if augment == 'common':
        if rotate > 0:
            trans += [
                Rotate(rotate, border_mode=0, p=p_rotate, interpolation=inter)
            ]
        if min_size == 1:
            trans += [Resize(size, size, interpolation=inter)]
        else:
            trans += [
                RandomResizedCrop(size,
                                  size,
                                  scale=(min_size, 1.0),
                                  p=1.0,
                                  interpolation=inter)
            ]
        trans += [HorizontalFlip(p=0.5)]
        if contrast > 0 or brightness > 0:
            trans += [RandomBrightnessContrast(brightness, contrast, p=0.5)]
        trans += [Normalize(MEAN, SD)]
    elif augment == 'eval':
        trans += [
            Resize(size, size, interpolation=inter),
            Normalize(MEAN, SD),
        ]
    else:
        raise NotImplementedError()

    trans += [GrayToTensor()]
    return Compose(trans)


def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


class GrayToTensor(ToTensorV2):
    def apply(self, img, **params):
        # because of gray scale
        # we add an additional channel
        return torch.from_numpy(img).unsqueeze(0)
