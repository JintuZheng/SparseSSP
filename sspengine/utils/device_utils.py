import torch

def data_transfer(data, target_device, keep_metas = False):
    new_data = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            new_data[k] = v.to(target_device)
        else:
            if keep_metas:
                new_data[k] = v
    return new_data
