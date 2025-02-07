import torch
from torch.utils.data import Dataset
from sspengine.engine import BUILDER
import torchvision.transforms as T
import os
import pickle

class AllencellDataset(Dataset):
    def __init__(self, data_path = None, one_file_data_path = None, adopted_sets_names = None, piplines = None):
        super().__init__()
        assert (data_path is not None and one_file_data_path is not None) == False, "`data_path` and `one_file_data_path` must exist only one!"
        if one_file_data_path is not None:
            data = torch.load(one_file_data_path)
            self.data = data['data']
            self.cache_mode = True
        else:
            assert data_path is not None, "`data_path` must be not None, if no `one_file_data_path` given"
            self.data = os.listdir(data_path)
            self.data_path = data_path
            self.cache_mode = False
        
        self.adopted_datasets = adopted_sets_names
        if piplines is not None:
            _pipelines = [BUILDER.build(p) for p in piplines]
            self.transform = T.Compose(_pipelines)
        else:
            self.transform = None
    
    def __len__(self, ):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.cache_mode:
            signal = self.data[index]['imgs'][0]
            target = self.data[index]['imgs'][1]
            info = self.data[index]['info']
            input_data = dict(
                img = signal,
                label = target,
                task_id = self.adopted_datasets.index(info['dataset']),
                # == img_metas ==
                set_name = info['dataset'],
                czi_path = info['path_czi'],
                src_shape = signal.shape
            )
        else:
            load_fname = os.path.join(self.data_path, self.data[index])
            with open(load_fname, 'rb') as f:
                input_data = pickle.load(f)
                input_data.update(load_fname = load_fname)

        if self.transform is not None:
            input_data = self.transform(input_data)

        return input_data

        