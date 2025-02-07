import os
import pickle
import argparse
from typing import OrderedDict
import torch
from sspengine.engine import Config, BUILDER
from tqdm import tqdm
from lightning.fabric import Fabric
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/repmode.py', help='path of configuration')
    parser.add_argument('--work_dirs_path', default='./work_dirs', help='path of logs')
    parser.add_argument('--checkpoint_path', default='./work_dirs/repmode/best.pth', help='path of checkpoint')
    parser.add_argument('--gpu_num', default = 4,)

    args = parser.parse_args()

    config_path = args.config
    exp_name = os.path.basename(config_path).split('.py')[0]
    path_log_dir = os.path.join(args.work_dirs_path, exp_name)
    path_cache_dir = os.path.join(args.work_dirs_path, exp_name, 'cache')

    if os.path.exists(path_cache_dir):
        shutil.rmtree(path_cache_dir)

    os.makedirs(path_cache_dir, exist_ok = True)
    os.makedirs(path_log_dir, exist_ok = True)

    cfg = Config.fromfile(args.config)

    # NOTE, Setup model
    model = BUILDER.build(cfg.model)
    state_dict = torch.load(args.checkpoint_path)
    model_state_dict = state_dict['model']

    new_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if 'bottle_neck' in k:
            new_dict[k.replace('bottle_neck', 'bottle')] = v
        else:
            new_dict[k] = v
    model_state_dict = new_dict

    model.load_state_dict(model_state_dict, strict = True)

    # NOTE, Setup dataloaders
    test_dataset = BUILDER.build(cfg.test_dataloader.dataset)
    cfg.test_dataloader.update(dict(dataset = test_dataset))
    test_dataloader = BUILDER.build(cfg.test_dataloader) # test

    # NOTE, Setup inference strategy
    infer_strategy = BUILDER.build(cfg.infer_strategy)

    # NOTE, Setup evaluation metric
    eval_metric = BUILDER.build(cfg.metric)

    fabric = Fabric(strategy = 'ddp', devices = args.gpu_num)
    fabric.launch()

    model = fabric.setup(model)
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    results = run_eval_test(model = model,
                dataloader = test_dataloader,
                infer_strategy = infer_strategy, 
                eval_metric = eval_metric,
                cfg = cfg)
    
    with open(os.path.join(path_cache_dir, f'_{str(fabric.global_rank)}_temp.pkl'), 'wb') as f:
        pickle.dump(results, f)
    fabric.barrier()
    if fabric.global_rank == 0:
        all_results = {}
        for i in range(fabric.world_size):
            with open(os.path.join(path_cache_dir, f'_{i}_temp.pkl'), 'rb') as f:
                _results = pickle.load(f)
                all_results.update(_results)
        all_results = list(all_results.values())
        metric_dict = eval_metric.reduce_make_logs(all_results, path_metric_save_dir = path_log_dir)
        print(metric_dict)
    
def run_eval_test(model, dataloader, infer_strategy, eval_metric, cfg):
    with torch.inference_mode():
        model.eval()
        results = {}
        for data in tqdm(dataloader):
            pred = infer_strategy(model, cfg.eval_batch_size_single_device, **data)
            target = data['label'].cpu()
            result_item = eval_metric(pred, target, set_name = data['set_name'], czi_path = data['czi_path'])
            results[data['load_fname'][0]] = result_item
        return results

if __name__ == '__main__':
    main()