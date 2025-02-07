import logging
import os
import datetime
import sys
import argparse
import torch
from sspengine.engine import Config, BUILDER
from sspengine.utils.time_utils import TrainingETA
from tqdm import tqdm
import numpy as np
from lightning.fabric import Fabric
import shutil
import pickle
import argparse

from lightning.fabric import Fabric

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default='./configs/sparse/repmode_sr8.py', help='path of configuration')
    parser.add_argument('--config', default='./configs/sparse/repmode_3e2d_sr8.py', help='path of configuration')
    parser.add_argument('--work_dirs_path', default='./work_dirs', help='path of logs')
    parser.add_argument('--seed', default = 1234,)
    parser.add_argument('--gpu_num', default = 4)

    args = parser.parse_args()

    config_path = args.config
    exp_name = os.path.basename(config_path).split('.py')[0]
    path_log_dir = os.path.join(args.work_dirs_path, exp_name)
    path_cache_dir = os.path.join(args.work_dirs_path, exp_name, 'cache')

    if os.path.exists(path_cache_dir):
        shutil.rmtree(path_cache_dir)
    os.makedirs(path_log_dir, exist_ok = True)
    os.makedirs(path_cache_dir, exist_ok = True)

    cfg = Config.fromfile(args.config)

    # NOTE, Setup model
    model = BUILDER.build(cfg.model)

    # NOTE, Setup loss func
    criterion = BUILDER.build(cfg.loss_func)

    # NOTE, Setup optimizer
    cfg.optimizer.update(dict(params = model.parameters()))
    optimizer = BUILDER.build(cfg.optimizer)

    # NOTE, Setup resume weight, if enable
    if cfg.resume_from_checkpoint_path is not None:
        state_dict = torch.load(cfg.resume_from_checkpoint_path)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])

    # NOTE, Setup dataloaders
    train_dataset = BUILDER.build(cfg.train_dataloader.dataset)
    cfg.train_dataloader.update(dict(dataset = train_dataset))
    train_dataloader = BUILDER.build(cfg.train_dataloader) # train 
    val_dataset = BUILDER.build(cfg.val_dataloader.dataset)
    cfg.val_dataloader.update(dict(dataset = val_dataset))
    val_dataloader = BUILDER.build(cfg.val_dataloader) # val

    # NOTE, Setup inference strategy
    infer_strategy = BUILDER.build(cfg.infer_strategy)

    # NOTE, Setup evaluation metric
    eval_metric = BUILDER.build(cfg.metric)

    fabric = Fabric(strategy = 'ddp', devices = args.gpu_num, precision = "16-mixed")
    fabric.launch()

    # NOTE, Setup the seed
    fabric.seed_everything(args.seed)

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    # NOTE, Setup lr_scheduler
    if cfg.lr_scheduler is not None:
        steps_each_train_dataloader = len(train_dataloader)
        milestones_epochs = cfg.lr_scheduler.milestones
        milestones = [steps_each_train_dataloader*mile for mile in milestones_epochs]
        cfg.lr_scheduler.update(dict(optimizer = optimizer, milestones = milestones))
        lr_scheduler = BUILDER.build(cfg.lr_scheduler)
    else:
        lr_scheduler = None

    # NOTE, Setup logging
    if fabric.global_rank == 0:
        logger = logging.getLogger('SSP_Engine')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(path_log_dir, f'train_{exp_name}.log'), mode='w')
        fh.setLevel(logging.DEBUG)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
        logger.addHandler(sh)
        logging.Formatter.converter = lambda a, b: (datetime.datetime.now() + datetime.timedelta(hours=8)).timetuple()
        logger.info(f'[DATASET] Adopted dataset: {str(cfg.adopted_sets_names)}')
        if 'sparse_ratio' in cfg.keys():
            logger.info(f'This is a sparse task training, the simulated sparse ratio is {cfg.sparse_ratio}')

    best_metric = np.inf

    eta_util = TrainingETA(total_steps = len(train_dataloader) * cfg.total_epochs)

    # Training & eval pipelines
    for epoch in range(cfg.total_epochs):

        # train
        for data in train_dataloader:
            model.train()
            optimizer.zero_grad()
            target_patch = data['label']
            with fabric.autocast():
                pred_patch = model(**data)
                loss_nomean = criterion(pred_patch, target_patch)
                loss = torch.mean(loss_nomean)
            fabric.backward(loss)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            eta_util.step()
            if fabric.global_rank == 0:
                logger.info('[TRAIN] [{}|{}] lr: {:.6f}, loss: {:.6f} {}'.format(epoch + 1, cfg.total_epochs, optimizer.param_groups[0]['lr'], loss.item(), eta_util.get_eta()))

        # eval
        if (epoch + 1) % cfg.eval_interval == 0:
            with torch.inference_mode():
                model.eval()
                results = {}
                for data in tqdm(val_dataloader):
                    pred = infer_strategy(model, cfg.eval_batch_size_single_device, **data)
                    target = data['label'].cpu()
                    result_item = eval_metric(pred, target, set_name = data['set_name'], czi_path = data['czi_path'])
                    results[data['load_fname'][0]] = result_item
            with open(os.path.join(path_cache_dir, f'_{str(fabric.global_rank)}_temp.pkl'), 'wb') as f:
                pickle.dump(results, f)
            fabric.barrier()
            all_results = {}
            for i in range(fabric.world_size):
                with open(os.path.join(path_cache_dir, f'_{i}_temp.pkl'), 'rb') as f:
                    _results = pickle.load(f)
                    all_results.update(_results)
            all_results = list(all_results.values())
            metric_dict = eval_metric.reduce_make_logs(all_results, path_metric_save_dir = None)

            if fabric.global_rank == 0:
                logger.info('[EVAL]  MSE: {:.6f}, MAE: {:.6f}, R2: {:.6f}'.format(metric_dict['all_MSE'], metric_dict['all_MAE'], metric_dict['all_R2']))

            # save the best model
            if metric_dict['all_MSE'] < best_metric:
                best_metric = metric_dict['all_MSE']
                path_best_model = os.path.join(path_log_dir, 'best.pth')
                state_dict = dict(
                    model = model.state_dict(),
                    optimizer = optimizer.state_dict(),
                    metas = dict(epoch = epoch)
                )
                fabric.save(path_best_model, state_dict)
                if fabric.global_rank == 0:
                    logger.info('[CKPT]   **Best** model saved to: {:s}'.format(path_best_model))
        
        
if __name__ == '__main__':
    main()