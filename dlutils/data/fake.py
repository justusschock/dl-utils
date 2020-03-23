from rising.loading import Dataset
import torch


__all__ = ['FakeDictDataset']

class FakeDictDataset(Dataset):
    def __init__(self, n_classes: int, shapes: dict, length=5000):
        super().__init__()
        self.n_classes = n_classes
        self.shapes = shapes
        self.length = length

    def __getitem__(self, item):
        return {k: torch.rand(shape) for k, shape in self.shapes.items()}

    def __len__(self):
        return self.length
