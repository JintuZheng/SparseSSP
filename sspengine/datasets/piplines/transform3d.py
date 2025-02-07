import numpy as np
import torch

class RandomCrop3D(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, input_data):
        signal = input_data['img']
        target = input_data['label']

        # random crop
        assert signal.shape == target.shape
        img_size = signal.shape[-3:]
        starts = np.array([
            np.random.randint(0, i - c + 1)
            for i, c in zip(img_size, self.patch_size)
        ])
        ends = starts + self.patch_size
        signal = signal[:, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
        target = target[:, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

        input_data['img'] = signal
        input_data['label'] = target

        return input_data
    
class RandomFilp3D(object):
    def __init__(self, random_flip_prob):
        self.random_flip_prob = random_flip_prob

    def __call__(self, input_data):

        signal = input_data['img']
        target = input_data['label']

        # random flip
        random_p = np.random.uniform(0, 1, size=3)
        filp_dims = list(np.where(random_p <= self.random_flip_prob)[0] + 1)
        signal = torch.flip(signal, dims=filp_dims)
        target = torch.flip(target, dims=filp_dims)

        input_data['img'] = signal
        input_data['label'] = target

        return input_data
        