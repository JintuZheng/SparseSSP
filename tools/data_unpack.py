from sspengine.datasets.allencell_dataset import AllencellDataset
from tqdm import tqdm
import os
import pickle
from multiprocessing import Process

one_file_data_folder = './data' # the folder stores the `.pth` file
target_unpacked_data_path = './data/unpacked' # the folder for each set.
splits = ['train', 'val', 'test']

adopted_sets_names = [
            'alpha_tubulin',
            'beta_actin',
            'desmoplakin',
            'dna',
            'fibrillarin',
            'lamin_b1',
            'membrane_caax_63x',
            'myosin_iib',
            'sec61_beta',
            'st6gal1',
            'tom20',
            'zo1',
        ]

for split_name in splits:
    id = 0
    print(f'Procssing {split_name}, loading dataset ...')
    dataset = AllencellDataset(one_file_data_path = os.path.join(one_file_data_folder, f'{split_name}.pth'), 
                               adopted_sets_names = adopted_sets_names)
    pkl_folder = os.path.join(target_unpacked_data_path, split_name)
    os.makedirs(pkl_folder, exist_ok = True)
    for data in tqdm(dataset):
        with open(os.path.join(pkl_folder, str(id) + '.pkl'), 'wb') as f:
            pickle.dump(data, f)
        id += 1