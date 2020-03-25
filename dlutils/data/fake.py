from typing import Union

import torch
from rising.loading import Dataset

__all__ = ['FakeDataset']


class FakeDataset(Dataset):
    def __init__(self, shapes: Union[dict, tuple, list],
                 length=5000):
        """
        Dataset generating fake data

        Parameters
        ----------
        shapes : dict or tuple or list
            a collection specifying the shapes to generate
        length : int
            length of the dataset
        """
        super().__init__()
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
