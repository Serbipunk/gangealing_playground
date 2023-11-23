from io import BytesIO
import lmdb
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])


class VideoDataset(Dataset):
    def __init__(self, path, transform=_transform, resolution=256, return_indices=False):
        self.capture = cv2.VideoCapture(path)
        assert self.capture.isOpened(), f'Failed to open video: {path}'
        # self.length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.length = 10

        self.resolution = resolution
        self.transform = transform if transform is not None else lambda x: x
        self.return_indices = return_indices

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # literally just read the next frame, index doesn't mean anything  # warning
        ret, frame = self.capture.read()
        if ret is False:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.capture.read()
            assert(ret is True)
        im_cv = cv2.resize(frame, (self.resolution, self.resolution))
        im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(im_cv)
        img = self.transform(im_pil)

        if self.return_indices:
            return img, index
        else:
            return img


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform=_transform, resolution=256, return_indices=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform if transform is not None else lambda x: x
        self.return_indices = return_indices

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        if self.return_indices:
            return img, index
        else:
            return img


def sample_infinite_data(loader, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    BIG_NUMBER = 9999999999999
    while True:
        # Randomize dataloader indices before every epoch:
        try:  # Only relevant for distributed sampler:
            shuffle_seed = torch.randint(0, BIG_NUMBER, (1,), generator=rng).item()
            loader.sampler.set_epoch(shuffle_seed)
        except AttributeError:
            pass
        for batch in loader:
            yield batch
