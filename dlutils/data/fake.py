from rising.loading import Dataset
import torch
from typing import Union


__all__ = ['FakeDataset']


class FakeDataset(Dataset):
    def __init__(self, n_classes: int, shapes: Union[dict, tuple, list],
                 length=5000):
        super().__init__()
        self.n_classes = n_classes
        self.shapes = shapes
        self.length = length

    def __getitem__(self, item):
        if isinstance(self.shapes, dict):
            return {k: torch.rand(shape) for k, shape in self.shapes.items()}
        elif isinstance(self.shapes, (tuple, list)):
            return type(self.shapes)([torch.rand(shape)
                                      for shape in self.shapes])

    def __len__(self):
        return self.length
