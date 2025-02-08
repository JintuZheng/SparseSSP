<div id="top" align="center">

# [ECCV 2024 (Oral)] SparseSSP: 3D Subcellular Structure Prediction from Sparse-View Transmitted Light Images

![](docs/assets/banner.jpg)

 SparseSSP uses less transmitted light (TL) images to obtain the high quality fluorescence results. **This repo provides a baseline implementation of sparse task on SSP. Welcome to improve it.**

</div>


<div align=center>

[![Slides](https://img.shields.io/badge/Slides-ECVA-blue?style=flat-square)](https://eccv2024.ecva.net/media/eccv-2024/Slides/874.pdf)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/abs/2407.02159)
[![Code](https://img.shields.io/badge/Code-Github-purple?style=flat-square)](https://github.com/JintuZheng/SparseSSP)

</div>


## Contents
- [Installation](#Installation)
- [Getting Started](#getting-started)
- [License & Citation](#license--citation)
- [Acknowledgement & Related Works](#acknowledgement--related-works)
- [Contact](#contact)

## Installation


```bash
pip3 install -r requirements.txt
python setup.py develop
```
> **NOTE**: Please check the version conflicts from your local env. We develop this code on `pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel`. It is recommand to use the same version if you are prefer to docker development.

## Getting Started

### STEP 1. Prepare Dataset

We are following the preprations from RepMode. Please refer to [here](https://github.com/Correr-Zhou/RepMode#-preparing-datasets) to download and make the formatted data files.

> ❗❗❗ Also, you can directly download the formatted data (*i.e.* the `.pth` files) [here. (link from RepMode)](https://1drv.ms/f/s!ArXcVhaRqzaZlo90CFbcJzvzu_izkw?e=Qo1ZeM)

Put all `.pth` files into the `./data`.

Check the paths in `tools/data_unpacked.py`.
```python
one_file_data_folder = './data' # the folder stores the `.pth` file
target_unpacked_data_path = './data/unpacked' # the folder for each set.
```

Then unpack all `.pth` files:
```bash
python ./tools/data_unpacked.py
```

The final path tree is like:
```
├─data
│  ├─unpacked
│  │  ├─test
│  │  ├─val
│  │  └─train
│  ├─train.pth
│  ├─val.pth
│  └─test.pth
```

### STEP 2. Train a Model

Write a configuration refer to the demo file `configs/sparse/repmode_3e2d_sr8.py`. Config is written on mmengine style.

> **Hints**: Demo config has no lr scheduler and but it supports the lr scheduler in `dist_train.py`.

Multiple GPUs training:

```bash
python tools/dist_train.py --config configs/sparse/repmode_3e2d_sr8.py --gpu_num 4
```

Single GPU training:

```bash
python tools/dist_train.py --config configs/sparse/repmode_3e2d_sr8.py --gpu_num 1
```

> **Hints**: Change the lr to `1e-4` may better if you want to use single gpu. The experments on papers are all in single gpu settings.

### STEP 3. Test a Model

> We provide a weight for easy test, and it is a `2000` epochs setting on sparse ratio 8. Download and put it on your code path. ([weight](https://drive.google.com/file/d/1OaQyfcy56JyHuoFJvOSuDE1vPTyVqE_s/view?usp=sharing) | [logs](https://drive.google.com/file/d/1rF2W5teULPzxqB8uV47Mx7ijl5QPTchW/view?usp=sharing))

Multiple GPUs testing: (4x faster)

```bash
python tools/dist_test.py --config configs/sparse/repmode_3e2d_sr8.py --gpu_num 4 --checkpoint_path ./demo.pth
```

Single GPU testing:

```bash
python tools/dist_test.py --config configs/sparse/repmode_3e2d_sr8.py --gpu_num 1 --checkpoint_path ./demo.pth
```

> **Hint**: The best checkpoint is under the `./work_dirs/xxx` if you follow STEP3 to train a model, fill it to `--checkpoint_path`.

<p align="right">(<a href="#top">back to top</a>)</p>


## License & Citation

If it is helpful to your research, please consider citing the following papers:

```bibtex

@inproceedings{zheng2024sparsessp,
    title={SparseSSP: 3D Subcellular Structure Prediction from Sparse-View Transmitted Light Images},
    author={Zheng, Jintu and Ding, Yi and Liu, Qizhe and Chen, Yuehui and Cao, Yi and Hu, Ying and Wang, Zenan},
    booktitle={European Conference on Computer Vision},
    pages={267--283},
    year={2024},
    organization={Springer}
}
@InProceedings{Zhou_2023_CVPR,
    author={Zhou, Donghao and Gu, Chunbin and Xu, Junde and Liu, Furui and Wang, Qiong and Chen, Guangyong and Heng, Pheng-Ann},
    title={RepMode: Learning to Re-Parameterize Diverse Experts for Subcellular Structure Prediction},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023},
    pages={3312-3322}
}
```

## Acknowledgement & Related Works

- **RepMode**, accepted to CVPR 2023 (Highlight): [offical repo](https://github.com/OpenDriveLab/DriveAGI) | [paper](https://arxiv.org/pdf/2212.10066.pdf)
- **FNet**, accpeted to Nature Methods: [offical repo](https://github.com/AllenCellModeling/pytorch_fnet/) | [paper](https://www.nature.com/articles/s41592-018-0111-2)

## Contact
Feel free to mailto zhengjintu22@mails.ucas.ac.cn if any questions. The `jt.zheng@siat.ac.cn` in paper is recycled from my org. now.

<p align="right">(<a href="#top">back to top</a>)</p>
