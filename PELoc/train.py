import io
import time
import pstats
import cProfile
import MinkowskiEngine as ME

from collections import OrderedDict
from omegaconf import OmegaConf, DictConfig
from pytorch3d.implicitron.tools import vis_utils
from accelerate import Accelerator
from utils.train_util import *

from datasets.base_loader import CollationFunctionFactory
from models.sgloc_v2 import Regressor
from models.sc2pcr import Matcher
from datasets.lidarloc import LiDARLocDataset
# from datasets.lidarloc_xac import LiDARLocDataset
from models.loss import CriterionCoordinate
from tqdm import tqdm
import math
from math import sin, cos, atan2, sqrt

import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def prefix_with_module(checkpoint):
    prefixed_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        prefixed_key = "module." + key
        prefixed_checkpoint[prefixed_key] = value
    return prefixed_checkpoint


# Wrapper for cProfile.Profile for easily make optional, turn on/off and printing
class Profiler:
    def __init__(self, active: bool):
        self.c_profiler = cProfile.Profile()
        self.active = active

    def enable(self):
        if self.active:
            self.c_profiler.enable()

    def disable(self):
        if self.active:
            self.c_profiler.disable()

    def print(self):
        if self.active:
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(self.c_profiler, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())


def get_thread_count(var_name):
    return os.environ.get(var_name)


def t_error(pred_poses, gt_poses):
    with torch.no_grad():
        error_t = val_translation(pred_poses, gt_poses)

    return error_t


def val_translation(pred_p, gt_p):
    """
    test model, compute error (numpy)
    input:
        pred_p: [3,]
        gt_p: [3,]
    returns:
        translation error (m):
    """
    pred_p = pred_p
    gt_p = gt_p.cpu().numpy()
    error = np.linalg.norm(gt_p - pred_p)

    return error


def train_fn(cfg: DictConfig):
    # NOTE carefully double check the instruction from huggingface!

    OmegaConf.set_struct(cfg, False)

    # Initialize the accelerator
    # accelerator = Accelerator(even_batches=False, device_placement=False, mixed_precision='fp16')
    
    accelerator = Accelerator(device_placement=False, mixed_precision='fp16')

    accelerator.print("Model Config:")
    accelerator.print(OmegaConf.to_yaml(cfg))

    accelerator.print(accelerator.state)

    torch.backends.cudnn.benchmark = cfg.TRAIN.cudnn_benchmark

    set_seed_and_print(cfg.TRAIN.seed)

    if accelerator.is_main_process:
        viz = vis_utils.get_visdom_connection(
            server="http://127.0.0.1",
            port=int(os.environ.get("VISDOM_PORT", 8097)),
        )

    accelerator.print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  OMP_NUM_THREADS: {get_thread_count('OMP_NUM_THREADS')}")
    accelerator.print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  MKL_NUM_THREADS: {get_thread_count('MKL_NUM_THREADS')}")

    accelerator.print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  SLURM_CPU_BIND: {get_thread_count('SLURM_CPU_BIND')}")
    accelerator.print(
        f"!!!!!!!!!!!!!!!!!!!!!!!!!!  SLURM_JOB_CPUS_PER_NODE: {get_thread_count('SLURM_JOB_CPUS_PER_NODE')}")

    data_set = cfg.TRAIN.data_set
    if data_set == 'XAC':
        train_dataset = LiDARLocDataset(
            root_dir='/home/data/CYD/data/XAC',  # which dataset?
            train=True,  # train or eval?
            voxel_size=0.4,  # voxel size of point cloud
            augment=False,  # use augmentation?
            aug_rotation=10,  # max rotation
            aug_translation=1,  # max translation
            generate_clusters=False,
            reg=True,
            level1_clusters=25,
            level2_clusters=100,
        )
    elif data_set == 'nuScenes':
        train_dataset = LiDARLocDataset(
            root_dir='/3D/lw/nuScenes',  # which dataset?
            train=True,  # train or eval?
            voxel_size=0.3,  # voxel size of point cloud
            augment=True,  # use augmentation?
            aug_rotation=10,  # max rotation
            aug_translation=1,  # max translation
            generate_clusters=False,
            reg=True,
            level1_clusters=25,
            level2_clusters=100,
        )
    elif data_set == 'KITTI':
        train_dataset = LiDARLocDataset(
            root_dir='/ava16t/lw/Data/KITTI',  # which dataset?
            train=True,  # train or eval?
            voxel_size=0.3,  # voxel size of point cloud
            augment=True,  # use augmentation?
            aug_rotation=10,  # max rotation
            aug_translation=1,  # max translation
            generate_clusters=False,
            reg=True,
            level1_clusters=25,
            level2_clusters=100,
        )
    elif data_set == 'QEOxford':
        train_dataset = LiDARLocDataset(
            root_dir='/home/data/CYD/data/Oxford',  # which dataset?
            train=True,  # train or eval?
            voxel_size=0.25,  # voxel size of point cloud
            augment=True,  # use augmentation?
            aug_rotation=10,  # max rotation
            aug_translation=1,  # max translation
            path = 3,
            scene = 'QEOxford',
            remove_points_ratio = 0.05 ### 每条轨迹随机去掉10% 5%的连续帧点云，防止依赖轨迹
        )
        
    elif data_set == 'Oxford':
            train_dataset = LiDARLocDataset(
            root_dir='/home/data/CYD/data/Oxford',  # which dataset?
            train=True,  # train or eval?
            voxel_size=0.25,  # voxel size of point cloud
            augment=True,  # use augmentation?
            aug_rotation=10,  # max rotation
            aug_translation=1,  # max translation
            path = 3,
            scene = 'Oxford',
            remove_points_ratio = 0.05 ### 每条轨迹随机去掉10% 5%的连续帧点云，防止依赖轨迹
        )
    elif data_set == 'NCLT':
        train_dataset = LiDARLocDataset(
            root_dir='/home/data/CYD/data/NCLT',  # which dataset?
            train=True,  # train or eval?
            voxel_size=0.25,  # voxel size of point cloud
            augment=True,  # use augmentation?
            aug_rotation=10,  # max rotation
            aug_translation=1,  # max translation
            path = 3,
            scene = 'NCLT',
            remove_points_ratio = 0.05 ### 每条轨迹随机去掉10% 5%的连续帧点云，防止依赖轨迹
        )
    else:
        accelerator.print(f"!!!!!!!!!!!!!!!!!!!!!!!!!! Only support Oxford or QEOxford")

    if cfg.TRAIN.num_workers > 0:
        persistent_workers = cfg.TRAIN.persistent_workers
    else:
        persistent_workers = False

    collation_fn = CollationFunctionFactory(collation_type='collate_pair_cls')

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.batch_size,
                                             num_workers=cfg.TRAIN.num_workers,
                                             pin_memory=cfg.TRAIN.pin_memory,
                                             shuffle=True, drop_last=True,
                                             persistent_workers=persistent_workers,
                                             collate_fn=collation_fn
                                             )  # collate_fn


    accelerator.print("length of train dataloader is: ", len(dataloader))

    # Instantiate the model
    ransac = Matcher(inlier_threshold=1.4)
    downpool = ME.MinkowskiAvgPooling(kernel_size=8, stride=8, dimension=3)


        
    model = Regressor(cfg.MODEL.num_head_blocks, cfg.MODEL.num_encoder_features, cfg.MODEL.mlp_ratio, reg=True)
    train_parameter = model.parameters()

    model = model.to(accelerator.device)
    accelerator.print('#Model parameters: {} M'.format(sum([x.nelement() for x in model.parameters()]) / 1e6))
    # accelerator.print('#Train parameters: {} M'.format(sum([x.nelement() for x in train_parameter()]) / 1e6))
    criterion = CriterionCoordinate()

    num_epochs = cfg.TRAIN.epochs

    # log
    if os.path.exists(cfg.TRAIN.exp_dir) == 0:
        os.mkdir(cfg.TRAIN.exp_dir)
    # Define the optimizer
    if cfg.TRAIN.warmup_sche:
        optimizer = torch.optim.AdamW(params=train_parameter, lr=cfg.TRAIN.lr)
        lr_scheduler = WarmupCosineLR(optimizer=optimizer, lr=cfg.TRAIN.lr,
                                      warmup_steps=cfg.TRAIN.restart_num * len(dataloader), momentum=0.9,
                                      max_steps=len(dataloader) * (cfg.TRAIN.epochs - cfg.TRAIN.restart_num))
  
    else:
        optimizer = torch.optim.AdamW(params=train_parameter, lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.weight_decay)
        lr_scheduler = CosineLR(optimizer, max_steps=len(dataloader) * (cfg.TRAIN.epochs))

    model, dataloader, optimizer, lr_scheduler = accelerator.prepare(model, dataloader, optimizer, lr_scheduler)

    accelerator.print(f"xxxxxxxxxxxxxxxxxx dataloader has {dataloader.num_workers} num_workers")

    start_epoch = 1

    to_plot = ("loss", "lr", "error_t")

    stats = VizStats(to_plot)

    for epoch in range(start_epoch, num_epochs + 1):
        stats.new_epoch()

        # # Evaluation
        if (epoch > 80) and (epoch % cfg.TRAIN.eval_interval == 0):
            # if (epoch%cfg.train.eval_interval ==0):
            # accelerator.print(f"----------Start to eval at epoch {epoch}----------")
            # _train_or_eval_fn(model, ransac, downpool, criterion, eval_dataloader, cfg, optimizer, stats, accelerator,
            #                   lr_scheduler, training=False)
            accelerator.print(f"----------Finish the eval at epoch {epoch}----------")
        else:
            accelerator.print(f"----------Skip the eval at epoch {epoch}----------")

        # Training
        accelerator.print(f"----------Start to train at epoch {epoch}----------")
        # 每轮训练前更新索引
        # data_set.update_index_list()    
        _train_or_eval_fn(model, ransac, downpool, criterion, dataloader, cfg, optimizer, stats, accelerator,
                          lr_scheduler, training=True)
        accelerator.print(f"----------Finish the train at epoch {epoch}----------")

        if accelerator.is_main_process:
            for g in optimizer.param_groups:
                lr = g['lr']
                break
            accelerator.print(f"----------LR is {lr}----------")
            accelerator.print(f"----------Saving stats to {cfg.TRAIN.exp_name}----------")
            stats.update({"lr": lr}, stat_set="train")
            stats.plot_stats(viz=viz, visdom_env=cfg.TRAIN.exp_name)
            accelerator.print(f"----------Done----------")

        # if epoch >= 0:
        if epoch >= 30 and epoch%5==0:
            accelerator.wait_for_everyone()
            ckpt_path = os.path.join(cfg.TRAIN.exp_dir, f"ckpt_{epoch:06}_{cfg.TRAIN.exp_name}.pth")
            accelerator.print(f"----------Saving the ckpt at epoch {epoch} to {ckpt_path}----------")
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), ckpt_path)

            if accelerator.is_main_process:
                stats.save(cfg.TRAIN.exp_dir + "stats")

    return True


def _train_or_eval_fn(model, ransac, downpool, criterion, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler,
                      training=True):
    if training:
        model.train()
    else:
        model.eval()

    # print(f"Start the loop for process {accelerator.process_index}")
    time_start = time.time()
    max_it = len(dataloader)

    tqdm_loader = tqdm(dataloader, total=len(dataloader))
    for step, batch in enumerate(tqdm_loader):
        
        features = batch['sinput_F'].to(accelerator.device, dtype=torch.float32)   ### (N,6)

        coordinates = batch['sinput_C'].to(accelerator.device)   ######### 第一维是batch中的索引号
        pcs_tensor = ME.SparseTensor(features[..., :3], coordinates)
        # breakpoint()
        pcs_tensor_s8 = ME.SparseTensor(features, coordinates)
        
        pose_gt = batch['pose'].to(accelerator.device, dtype=torch.float32)

        batch_size = pose_gt.size(0)
        pred_t = np.zeros((batch_size, 3))
        pred_q = np.zeros((batch_size, 4))
        index_list = [0]  # 用于存放索引
        with torch.no_grad():
            ground_truth = downpool(pcs_tensor_s8)

        if training:
         
            predictions = model(pcs_tensor)      ##### 列表
            pred = predictions['pred']
            features_512 = predictions['f_512'].F

            ground_truth = ground_truth.features_at_coordinates(pred.C.float())

            gt_sup_point = ground_truth[:, 3:6]
            # mask = ground_truth[:, 6].view(-1, 1)

            pred_point = pred.F
            predictions['loss'] = criterion(pred_point, gt_sup_point,features_512) 
            loss = predictions['loss']
        else:
            with torch.no_grad():
                predictions = model(pcs_tensor)
            pred = predictions['pred']
            pred_point = pred.F
            ground_truth = ground_truth.features_at_coordinates(pred.C.float())
            sup_point = ground_truth[:, :3]
            for i in range(batch_size):
                # 取出预测的每个batch中的坐标点
                batch_pred_pcs_tensor = pred.coordinates_at(i).float()
                index_list.append(index_list[i] + len(batch_pred_pcs_tensor))
            gt_point = sup_point
            for i in range(batch_size):
                a = gt_point[index_list[i]:index_list[i + 1], :]
                b = pred_point[index_list[i]:index_list[i + 1], :]
                batch_pred_t, batch_pred_q = ransac.estimator(
                    a.unsqueeze(0), b.unsqueeze(0))
                pred_t[i, :] = batch_pred_t
                pred_q[i, :] = batch_pred_q
                if i == 0:
                    error_t = t_error(pred_t[i, :3], pose_gt[i, :3])
                else:
                    error_t += (t_error(pred_t[i, :3], pose_gt[i, :3]))

            predictions['error_t'] = error_t / batch_size

        if training:
            stats.update(predictions, time_start=time_start, stat_set="train")
            if step % cfg.TRAIN.print_interval == 0:
                accelerator.print(stats.print(stat_set="train", max_it=max_it))
        else:
            stats.update(predictions, time_start=time_start, stat_set="eval")
            if step % cfg.TRAIN.print_interval == 0:
                accelerator.print(stats.print(stat_set="eval", max_it=max_it))

        if training:
            optimizer.zero_grad()
            accelerator.backward(loss)
   
            optimizer.step()
            lr_scheduler.step()
    # 清空显存
    torch.cuda.empty_cache()
    return True


if __name__ == '__main__':
    ## use different config
    conf = OmegaConf.load('config/qeoxford.yaml')
    # conf = OmegaConf.load('config/oxford.yaml')
    # conf = OmegaConf.load('config/nclt.yaml')
    # conf = OmegaConf.load('config/xac.yaml')
    train_fn(conf)
