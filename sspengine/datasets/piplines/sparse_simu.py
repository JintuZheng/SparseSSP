import torch
import numpy as np

class PatchSparseSimulation(object):
    def __init__(self, sparse_ratio):
        self.sparse_ratio = sparse_ratio

    def __call__(self, input_data):
        signal = input_data['img']

        signal = signal[:, ::self.sparse_ratio, :, :]

        input_data['img'] = signal

        return input_data