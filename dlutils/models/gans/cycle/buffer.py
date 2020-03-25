import numpy as np
import torch


class ReplayBuffer(object):
    """
    A replay Buffer which returns old samples with a probability of 0.5 instead
    of new samples and enqueues the new samples for later usage

    """

    def __init__(self, max_size=50, to_cpu=True):
        """

        Parameters
        ----------
        max_size : int
            the maximum number of samples to enqueue
        to_cpu : bool
            whether to store the enqueued values on cpu

        Notes
        -----
        Storing intermediate values on CPU may increase train time, since
        additional device transfers must be made, but the replay buffer won't
        use GPU RAM. If disabling this option, all tensors live in GPU RAM,
        which might easily cause ``OutOfMemory`` Errors
        """
        self.max_size = max_size
        self.to_cpu = to_cpu

        self.data = []

    def __call__(self, batch: torch.Tensor):
        """
        Makes the class callable. If called, an old sample is returned instead
        of the current sample with a probability of 50% for each sample in the
        current batch

        Parameters
        ----------
        batch : :class:`torch.Tensor`
            the current batch

        Returns
        -------
        :class:`torch.Tensor`
            the returned batch

        """
        return_batch = []

        for img in batch.detach():
            if self.to_cpu:
                img = img.to("cpu")
            img = torch.unsqueeze(img, 0)

            if len(self.data) < self.max_size:
                self.data.append(img)
                return_batch.append(img)

            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size - 1)

                    return_batch.append(self.data[i].clone())
                    self.data[i] = img

                else:
                    return_batch.append(img)

        return torch.cat(return_batch).to(batch.device)
