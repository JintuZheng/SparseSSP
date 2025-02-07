from sspengine.models.task_unet_predictor import TaskUNetPredictor
from sspengine.utils.task_embedding_funcs import repmode_task_embedding_func
from sspengine.modules.MoDE.layers import MoDEEncoderLayers, MoDEDecoderLayers, MoDEEncoderBlock, MoDEDecoderBlock
from sspengine.modules.MoDE.commons import MoDESubNet2Conv, MoDEConv

from sspengine.utils.sparse_ssp_utils import PrefixInterpolation, DepthChannelSwitcher

from sspengine.datasets import AllencellDataset
from sspengine.datasets.piplines import RandomCrop3D, RandomFilp3D, PatchSparseSimulation

from sspengine.utils.inference_strategy import GaussianWindowSliding

from sspengine.utils.metrics import BaseMetric

from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

sparse_ratio = 8

num_tasks = 12
num_experts = 5
patch_size = (32, 128, 128)
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

# ==========================================
# STEP1. Define model
# ==========================================

model = dict(
    type = TaskUNetPredictor,
    task_emb_func = repmode_task_embedding_func,
    # NOTE, voxel prefix interp.
    encoder_before_hook = dict(
        type = PrefixInterpolation,
        target_voxel_size = patch_size,
    ),
    # NOTE, 3d encoder
    encoder_layers = dict(
        type = MoDEEncoderLayers,
        block = MoDEEncoderBlock, 
        num_blocks = 4, 
        num_experts = num_experts, 
        num_tasks = num_tasks, 
        in_channels = 1, 
        out_channels_list = [32, 64, 128, 256],
    ),
    bottle_before_hook = None,
    bottle = dict(
        type = MoDESubNet2Conv,
        num_experts = num_experts,
        num_tasks = num_tasks, 
        n_in = 256, 
        n_out = 512,
    ),
    # NOTE, depth to channels
    decoder_before_hook = dict(
        type = DepthChannelSwitcher,
        target_voxel_size = patch_size,
        gate_in_2d_channels = 1024, # in 3d encoder, depths would change to [32 // 2, 32 // 4, 32 // 8, 32 // 16], so 1024 = 512 * 2
        gate_out_2d_channels_list = [64, 128, 256, 512],
        out_2d_gate = dict(
            type = MoDEConv,
            num_experts = num_experts, 
            num_tasks = num_tasks, 
            kernel_size = 5, 
            using_2d_ops = True,
        ),
        mode = 'depths_to_chans'
    ),
    # NOTE, 2d decoder
    decoder_layers = dict(
        type = MoDEDecoderLayers,
        block = MoDEDecoderBlock, 
        num_blocks = 4, 
        num_experts = num_experts,
        num_tasks = num_tasks, 
        in_channels = 1024, 
        out_channels_list = [512, 256, 128, 64],
        using_2d_ops = True, # enable this would use the 2d ops.
    ),
    out_head_before_hook = None,
    # NOTE, 2d out head
    out_head = dict(
        type = MoDEConv,
        num_experts = num_experts, 
        num_tasks = num_tasks, 
        in_chan = 64,
        out_chan = 32,
        kernel_size = 5, 
        conv_type = 'final',
        using_2d_ops = True, # enable this would use the 2d ops.
    ),
    # NOTE, restore to 3d
    out_head_after_hook = dict(
        type = DepthChannelSwitcher,
        target_voxel_size = patch_size,
        mode = 'chans_to_depths'
    ),
)

infer_strategy = dict(
    type = GaussianWindowSliding,
    patch_size = patch_size,
    sparse_ratio = sparse_ratio,
)

# ==========================================
# STEP2. Define datasets & dataloader
# ==========================================

batch_size_each_device = 8
eval_batch_size_single_device = 8

train_dataset = dict(
    type = AllencellDataset,
    # one_file_data_path = 'data/train.pth', # less IO time but cost at least `64 * gpu_num`GB memory
    data_path = 'data/unpacked/train', 
    adopted_sets_names = adopted_sets_names,
    piplines = [
        dict(type = RandomCrop3D, patch_size = patch_size),
        dict(type = RandomFilp3D, random_flip_prob = 0.5),
        dict(type = PatchSparseSimulation, sparse_ratio = sparse_ratio)
    ]
)
train_dataloader = dict(
    type = DataLoader,
    dataset = train_dataset,
    batch_size = batch_size_each_device,
    shuffle = True,
    num_workers = 10,
    pin_memory = True,
    persistent_workers = True,
)

val_dataset = dict(
    type = AllencellDataset,
    # one_file_data_path = 'data/val.pth', # less IO time but cost at least `64 * gpu_num`GB memory
    data_path = 'data/unpacked/val', 
    adopted_sets_names = adopted_sets_names,
)
val_dataloader = dict(
    type = DataLoader,
    dataset = val_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 10,
    pin_memory= True,
    persistent_workers = True,
)


test_dataset = dict(
    type = AllencellDataset,
    # one_file_data_path = 'data/test.pth', # less IO time but cost at least `64 * gpu_num`GB memory
    data_path = 'data/unpacked/test', 
    adopted_sets_names = adopted_sets_names,
)
test_dataloader = dict(
    type = DataLoader,
    dataset = test_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 10,
    pin_memory = True,
    persistent_workers = True,
)

metric = dict(
    type = BaseMetric,
    return_keys = ['MSE', 'MAE', 'R2'],
    return_pd_format = True,
)

# ==========================================
# STEP3. Define training parameters
# ==========================================

total_epochs = 2000
eval_interval = 10

loss_func = dict(
    type = MSELoss,
    reduction = 'none',
)
optimizer = dict(
    type = Adam,
    lr = 4e-4,
)
lr_scheduler = None
resume_from_checkpoint_path = None
