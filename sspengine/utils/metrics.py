import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics
import torch
import pandas as pd
import os

class BaseMetric(object):
    def __init__(self, return_keys, return_pd_format = True) -> None:
        self.return_keys = return_keys
        self.return_pd_format = return_pd_format
    
    def convert_pd_format(self, result:dict, set_name, czi_path):
        metric_sample = pd.DataFrame([result])
        metric_sample.insert(loc=0, column='set_name', value = set_name)
        metric_sample.insert(loc=1, column='czi_path', value = czi_path)
        return metric_sample
    
    def reduce_make_logs(self, results:list, path_metric_save_dir:str = None):
        comp_metric_df = pd.concat(results)
        id_data = ['{:0>3d}'.format(i) for i in range(len(comp_metric_df))]
        comp_metric_df.insert(loc=2, column='id', value=id_data)
        spec_metric_df = comp_metric_df.groupby('set_name').mean(numeric_only=True)
        final_metric_df = comp_metric_df.mean(numeric_only=True).to_frame().T

        log_dict = {}
        for column in final_metric_df:
            log_dict[f'all_{column}'] = final_metric_df.iloc[0][column]
            for index in spec_metric_df.index:
                log_dict[f'{column}/{index}'] = spec_metric_df.loc[index][column]

        if path_metric_save_dir is not None:
            spec_metric_df.insert(loc=0, column='set_name', value = spec_metric_df.index)
            spec_metric_df.reset_index(drop=True, inplace=True)
            comp_metric_df.to_csv(os.path.join(path_metric_save_dir, 'comp.csv'), index=False)
            spec_metric_df.to_csv(os.path.join(path_metric_save_dir, 'spec.csv'), index=False)
            final_metric_df.to_csv(os.path.join(path_metric_save_dir, 'final.csv'), index=False)
            
        return log_dict

    
    def __call__(self, pred, target, **kargs):
        pred = pred.unsqueeze(0).unsqueeze(0).numpy()
        target = target.unsqueeze(0).unsqueeze(0).numpy()
        target_flat = target.flatten()
        pred_flat = pred.flatten()

        err_map = np.abs(pred - target) if 'err_map' in self.return_keys else 0.
        MSE = metrics.mean_squared_error(target_flat, pred_flat) if 'MSE' in self.return_keys else 0.
        MAE = metrics.mean_absolute_error(target_flat, pred_flat) if 'MAE' in self.return_keys else 0.
        R2 = metrics.r2_score(target_flat, pred_flat) if 'R2' in self.return_keys else 0.

        metric_result = {
            'MSE': MSE,
            'MAE': MAE,
            'R2': R2,
            'err_map': err_map,
        }

        if self.return_pd_format:
            return self.convert_pd_format(metric_result, **kargs)
        else:
            return metric_result
