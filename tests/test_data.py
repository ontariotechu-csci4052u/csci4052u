import pytest
from csci4052u.data import FashionMNISTDataModule


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("data"))


class TestFractionSampling:
    def test_full_fraction(self, data_dir):
        dm = FashionMNISTDataModule(data_dir=data_dir, fraction=1.0)
        dm.prepare_data()
        dm.setup()
        assert len(dm.train_dataset) + len(dm.val_dataset) == 60000
        assert len(dm.test_dataset) == 10000

    def test_half_fraction(self, data_dir):
        dm = FashionMNISTDataModule(data_dir=data_dir, fraction=0.5)
        dm.prepare_data()
        dm.setup()
        assert len(dm.train_dataset) + len(dm.val_dataset) == 30000
        assert len(dm.test_dataset) == 5000

    def test_small_fraction(self, data_dir):
        dm = FashionMNISTDataModule(data_dir=data_dir, fraction=0.01)
        dm.prepare_data()
        dm.setup()
        assert len(dm.train_dataset) + len(dm.val_dataset) == 600
        assert len(dm.test_dataset) == 100

    def test_train_val_ratio(self, data_dir):
        dm = FashionMNISTDataModule(data_dir=data_dir, fraction=0.1)
        dm.prepare_data()
        dm.setup()
        total = len(dm.train_dataset) + len(dm.val_dataset)
        assert total == 6000
        assert len(dm.train_dataset) == int(total * 5 / 6)


class TestShapeTransform:
    def test_default_shape(self, data_dir):
        dm = FashionMNISTDataModule(data_dir=data_dir, fraction=0.01)
        dm.prepare_data()
        dm.setup()
        img, _ = dm.train_dataset[0]
        assert img.shape == (1, 28, 28)

    def test_resize(self, data_dir):
        dm = FashionMNISTDataModule(data_dir=data_dir, fraction=0.01, shape=(1, 32, 32))
        dm.prepare_data()
        dm.setup()
        img, _ = dm.train_dataset[0]
        assert img.shape == (1, 32, 32)

    def test_expand_to_3_channels(self, data_dir):
        dm = FashionMNISTDataModule(data_dir=data_dir, fraction=0.01, shape=(3, 28, 28))
        dm.prepare_data()
        dm.setup()
        img, _ = dm.train_dataset[0]
        assert img.shape == (3, 28, 28)

    def test_resize_and_expand(self, data_dir):
        dm = FashionMNISTDataModule(data_dir=data_dir, fraction=0.01, shape=(3, 64, 64))
        dm.prepare_data()
        dm.setup()
        img, _ = dm.train_dataset[0]
        assert img.shape == (3, 64, 64)
