import math
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter

class GaussianWindowSliding(object):
    def __init__(self, patch_size, sparse_ratio) -> None:
        self.patch_size = patch_size
        self.sparse_ratio = sparse_ratio
    
    def get_gaussian(self, sigma_scale = 1 / 8):
        tmp = np.zeros(self.patch_size)
        center_coords = [i // 2 for i in self.patch_size]
        sigmas = [i * sigma_scale for i in self.patch_size]
        tmp[tuple(center_coords)] = 1
        gauss_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gauss_map = gauss_map / np.max(gauss_map) * 1
        gauss_map = gauss_map.astype(np.float32)
        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gauss_map[gauss_map == 0] = np.min(gauss_map[gauss_map != 0])
        return gauss_map

    def __call__(self, model, batch_size_eval, img, task_id, **kargs):
        img_size = img.shape[-3:]
        over_lap_ratio = 0.5
        strides = [
            int(math.ceil(patch_len * (1 - over_lap_ratio)))
            for patch_len in self.patch_size
        ]
        steps = [
            int(math.ceil((img_len - patch_len) / stride + 1))
            for img_len, patch_len, stride in zip(img_size, self.patch_size, strides)
        ]

        gauss_map = self.get_gaussian()
        gauss_map = torch.from_numpy(gauss_map).to(img.device)
        pred_sum = torch.zeros(img.shape, device = img.device)
        weight_sum = torch.zeros(img.shape, device = img.device)

        # obtain patchs of signal img
        patchs = list()
        for i in range(steps[0]):
            for j in range(steps[1]):
                for k in range(steps[2]):
                    indexs = [i, j, k]
                    starts = [
                        int(idx * stride)
                        for idx, stride in zip(indexs, strides)
                    ]
                    ends = [  # prevent overflow
                        min(start + patch_len, img_len)
                        for start, patch_len, img_len in zip(starts, self.patch_size, img_size)
                    ]
                    starts = [  # readjust starts
                        max(int(end - patch_len), 0)
                        for end, patch_len in zip(ends, self.patch_size)
                    ]
                    patch_img = img[:, :, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
                    
                    if self.sparse_ratio is not None:
                        # TODO, a real sparse impl.
                        patch_img = patch_img[:, :, ::self.sparse_ratio, :, :] # now it is an inter-patch sparse simulation.
                    patchs.append({
                        'starts_zyx': starts,
                        'ends_zyx': ends,
                        'patch': patch_img,
                    })

        batch_buffer = list()
        while True:
            batch_buffer.append(patchs.pop())
            if len(batch_buffer) == batch_size_eval or len(patchs) == 0:

                # bulid batch input
                signal_patchs = torch.cat([element['patch'] for element in batch_buffer], dim=0)

                # predict
                with torch.no_grad():
                    pred_patchs = model(signal_patchs, task_id.expand(len(batch_buffer)))
                    if isinstance(pred_patchs, tuple):
                        pred_patchs = pred_patchs[0]

                # consolidate
                for i, element in enumerate(batch_buffer):
                    starts = element['starts_zyx']
                    ends = element['ends_zyx']
                    pred_patch = pred_patchs[i].unsqueeze(0)
                    pred_sum[:, :, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]] += pred_patch[:, :] * gauss_map
                    weight_sum[:, :, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]] += gauss_map

                # clear buffer
                batch_buffer.clear()
                if len(patchs) == 0: break

        # get final prediction
        pred = pred_sum / weight_sum
        return pred.cpu()