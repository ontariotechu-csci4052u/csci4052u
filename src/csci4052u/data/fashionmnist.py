import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST


class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=64, num_workers=2, shape=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        t = [transforms.ToTensor()]
        if shape is not None:
            channels, h, w = shape
            t.append(transforms.Resize((h, w)))
            if channels == 3:
                t.append(transforms.Lambda(lambda x: x.expand(3, -1, -1)))
            t.append(transforms.Normalize((0.5,) * channels, (0.5,) * channels))
        else:
            t.append(transforms.Normalize((0.5,), (0.5,)))
        self.transform = transforms.Compose(t)

    def prepare_data(self):
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_train = FashionMNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.train_dataset, self.val_dataset = random_split(
                full_train, [50000, 10000]
            )
        if stage == "test" or stage is None:
            self.test_dataset = FashionMNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
