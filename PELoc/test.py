import time
import MinkowskiEngine as ME
import matplotlib
import numpy as np
from tqdm import tqdm
from utils.train_util import *
# from datasets.single_nuscenes import Nuscenes
from datasets.base_loader import CollationFunctionFactory
from utils.pose_util import val_translation, val_rotation, qexp, estimate_pose
from models.sgloc_v2 import Regressor
from models.sc2pcr import Matcher
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from os import path as osp
from tensorboardX import SummaryWriter
from datasets.lidarloc import LiDARLocDataset
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def test(cfg: DictConfig):

    global TOTAL_ITERATIONS
    TOTAL_ITERATIONS = 0
    OmegaConf.set_struct(cfg, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Regressor(cfg.MODEL.num_head_blocks, cfg.MODEL.num_encoder_features, cfg.MODEL.mlp_ratio, reg=True)
    ransac = Matcher(inlier_threshold=1.4)
    downpool = ME.MinkowskiAvgPooling(kernel_size=8, stride=8, dimension=3)
    model.to(device)

    if cfg.TRAIN.data_set == 'XAC':
        eval_dataset = LiDARLocDataset(
            root_dir='/home/data/CYD/data/XAC',  # which dataset?
            train=False,  # train or eval?
            voxel_size=0.4,  # voxel size of point cloud
            augment=True,  # use augmentation?
            aug_rotation=10,  # max rotation
            aug_translation=1,  # max translation
            generate_clusters=False,
            reg=True,
            level1_clusters=25,
            level2_clusters=100,
        )
        pose_stats = os.path.join(cfg.TRAIN.data_root, 'XAC_pose_stats.txt')
    elif cfg.TRAIN.data_set == 'NCLT':
        eval_dataset = LiDARLocDataset(
            root_dir='/home/data/CYD/data/NCLT',  # which dataset?
            train=False,  # train or eval?
            voxel_size=0.25,  # voxel size of point cloud
            augment=False,  # use augmentation?
            aug_rotation=10,  # max rotation
            aug_translation=1,  # max translation
            path = 3,
            scene = 'NCLT',
            remove_points_ratio = 0.
        )
        # pose_stats = os.path.join(cfg.TRAIN.data_root, 'NCLT', 'NCLT_pose_stats.txt')
        pose_stats = os.path.join(cfg.TRAIN.data_root, 'NCLT_pose_stats.txt')
        
    elif cfg.TRAIN.data_set == 'Oxford':
        eval_dataset = LiDARLocDataset(
            root_dir='/home/data/CYD/data/Oxford',  # which dataset?
            train=False,  # train or eval?
            voxel_size=0.25,  # voxel size of point cloud
            augment=False,  # use augmentation?
            aug_rotation=10,  # max rotation
            aug_translation=1,  # max translation
            path = 3,
            scene = 'Oxford',
            remove_points_ratio = 0.
        )
        # pose_stats = os.path.join(cfg.TRAIN.data_root, 'Oxford', 'Oxford_pose_stats.txt')
        pose_stats = os.path.join(cfg.TRAIN.data_root, 'Oxford_pose_stats.txt')

    elif cfg.TRAIN.data_set == 'QEOxford':   ### cfg.data_set
        eval_dataset = LiDARLocDataset(
            root_dir='/home/data/CYD/data/Oxford',  # which dataset?
            train=False,  # train or eval?
            voxel_size=0.25,  # voxel size of point cloud
            augment=False,  # use augmentation?
            aug_rotation=10,  # max rotation
            aug_translation=1,  # max translation
            path = 3,
            scene = 'QEOxford',
            remove_points_ratio = 0. ### we only use RFT in training
        )
        # pose_stats = os.path.join(cfg.TRAIN.data_root, 'Oxford', 'QEOxford_pose_stats.txt')
        pose_stats = os.path.join(cfg.TRAIN.data_root, 'QEOxford_pose_stats.txt')
    elif cfg.TRAIN.data_set == 'KITTI':
        eval_dataset = LiDARLocDataset(
            root_dir='/ava16t/lw/Data/KITTI',  # which dataset?
            train=False,  # train or eval?
            voxel_size=0.3,  # voxel size of point cloud
            augment=False,  # use augmentation?
            aug_rotation=10,  # max rotation
            aug_translation=1,  # max translation
            generate_clusters=False,
            reg=True,
            level1_clusters=25,
            level2_clusters=100,
        )
        pose_stats = os.path.join(cfg.TRAIN.data_root, 'KITTI', 'KITTI_pose_stats.txt')


    print('#Model parameters: {} M'.format(sum([x.nelement() for x in model.parameters()])/1e6))
    print('#Regressor parameters: {} M'.format(sum([x.nelement() for x in model.reg_heads.parameters()])/1e6) )

    if cfg.TRAIN.num_workers > 0:
        persistent_workers = cfg.TRAIN.persistent_workers
    else:
        persistent_workers = False

    collation_fn = CollationFunctionFactory(collation_type='collate_pair_cls')

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.TRAIN.batch_size,
                                                  num_workers=cfg.TRAIN.num_workers, pin_memory=cfg.TRAIN.pin_memory,
                                                  shuffle=False, persistent_workers=persistent_workers,
                                                  collate_fn=collation_fn)  # collate_fn

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()
    for epoch in range(45, 46, 1):
        log_string('**** EPOCH %03d ****' % epoch)
        # ckpt_path = osp.join(cfg.TRAIN.exp_dir, 'ckpt_' + str(epoch).zfill(6) + '.pth')
        # ckpt_path = osp.join(cfg.TRAIN.exp_dir, 'ckpt_000100_Ouster.pth')
        # ckpt_path = osp.join('log/', 'Encoder/single_KITTI.pth')
        # ckpt_path = osp.join('log/', '15-13-06-37/ckpt_000050_QEOxford.pth')
        # ckpt_path = osp.join('log/', '15-13-06-37/ckpt_000045_Oxford.pth')
        ckpt_path = osp.join('log/', '2012-05-26/ckpt_000050_NCLT.pth')
        print(ckpt_path)
        # exit()
        if osp.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint, strict=True)
            # model.load_state_dict(checkpoint.get("model_state_dict", {}))
            print(f"Loaded checkpoint from: {ckpt_path}")
        else:
            raise ValueError(f"No checkpoint found at: {ckpt_path}")

        model.eval()

        # Seed random engines
        seed_all_random_engines(cfg.TRAIN.seed)

        # pose mean and std
        if cfg.TRAIN.data_set != 'Nuscenes':
            pose_m = np.loadtxt(pose_stats)
        else:
            pose_m = np.array([0, 0, 0])

        gt_translation = np.zeros((len(eval_dataset), 3))
        pred_translation = np.zeros((len(eval_dataset), 3))
        gt_rotation = np.zeros((len(eval_dataset), 4))
        pred_rotation = np.zeros((len(eval_dataset), 4))

        error_t = np.zeros(len(eval_dataset))
        error_txy = np.zeros(len(eval_dataset))
        error_q = np.zeros(len(eval_dataset))

        time_results_network = []
        time_results_ransac = []
        num = 0
        tqdm_loader = tqdm(eval_dataloader, total=len(eval_dataloader))
        for step, batch in enumerate(tqdm_loader):
            val_pose = batch["pose"]
            start_idx = step * cfg.TRAIN.batch_size
            end_idx = min((step + 1) * cfg.TRAIN.batch_size, len(eval_dataset))
            gt_translation[start_idx:end_idx, :] = val_pose[:, :3].numpy() + pose_m
            gt_rotation[start_idx:end_idx, :] = np.asarray([qexp(q) for q in val_pose[:, 3:].numpy()])

            features = batch['sinput_F'].to(device, dtype=torch.float32)
            coordinates = batch['sinput_C'].to(device)
            pcs_tensor = ME.SparseTensor(features[..., :3], coordinates)

            pcs_tensor_s8 = ME.SparseTensor(features, coordinates)
            # 原始点数的GT
            pose_gt = batch['pose'].to(device, dtype=torch.float32)
            batch_size = pose_gt.size(0)
            pred_t = np.zeros((batch_size, 3))
            pred_q = np.zeros((batch_size, 4))
            index_list = [0]  # 用于存放索引
            start = time.time()
            with torch.no_grad():
                ground_truth = downpool(pcs_tensor_s8)
                predictions = model(pcs_tensor)
            end = time.time()
            cost_time = (end - start) / batch_size
            time_results_network.append(cost_time)

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
                batch_pred_t, batch_pred_q, _ = ransac.estimator(
                    a.unsqueeze(0), b.unsqueeze(0))
                pred_t[i, :] = batch_pred_t
                pred_q[i, :] = batch_pred_q
                
            end = time.time()
            cost_time = (end - start) / batch_size
            time_results_ransac.append(cost_time)

            pred_translation[start_idx:end_idx, :] = pred_t + pose_m

            pred_rotation[start_idx:end_idx, :] = pred_q
            num +=1
            error_t[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in
                                                     zip(pred_translation[start_idx:end_idx, :],
                                                         gt_translation[start_idx:end_idx, :])])
            error_txy[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in
                                                       zip(pred_translation[start_idx:end_idx, :2],
                                                           gt_translation[start_idx:end_idx, :2])])

            error_q[start_idx:end_idx] = np.asarray(
                [val_rotation(p, q) for p, q in zip(pred_rotation[start_idx:end_idx, :],
                                                    gt_rotation[start_idx:end_idx, :])])

            # log_string('ValLoss(m): %f' % float(val_loss))
            log_string('MeanXYZTE(m): %f' % np.mean(error_t[start_idx:end_idx], axis=0))
            log_string('MeanXYTE(m): %f' % np.mean(error_txy[start_idx:end_idx], axis=0))
            log_string('MeanRE(degrees): %f' % np.mean(error_q[start_idx:end_idx], axis=0))
            log_string('MedianTE(m): %f' % np.median(error_t[start_idx:end_idx], axis=0))
            log_string('MedianRE(degrees): %f' % np.median(error_q[start_idx:end_idx], axis=0))

            torch.cuda.empty_cache()
        mean_ATE = np.mean(error_t)
        mean_xyATE = np.mean(error_txy)
        mean_ARE = np.mean(error_q)
        median_ATE = np.median(error_t)
        median_xyATE = np.median(error_txy)
        median_ARE = np.median(error_q)
        mean_time_network = np.mean(time_results_network)
        mean_time_ransac = np.mean(time_results_ransac)
        log_string('Mean Position Error(m): %f' % mean_ATE)
        log_string('Mean XY Position Error(m): %f' % mean_xyATE)
        log_string('Mean Orientation Error(degrees): %f' % mean_ARE)
        log_string('Median Position Error(m): %f' % median_ATE)
        log_string('Median XY Position Error(m): %f' % median_xyATE)
        log_string('Median Orientation Error(degrees): %f' % median_ARE)
        log_string('Mean Network Cost Time(s): %f' % mean_time_network)
        log_string('Mean Ransac Cost Time(s): %f' % mean_time_ransac)
        val_writer.add_scalar('MeanATE', mean_ATE, TOTAL_ITERATIONS)
        val_writer.add_scalar('MeanARE', mean_ARE, TOTAL_ITERATIONS)

        # save error
        error_t_filename = osp.join(cfg.TRAIN.exp_dir, 'error_t.txt')
        error_q_filename = osp.join(cfg.TRAIN.exp_dir, 'error_q.txt')
        np.savetxt(error_t_filename, error_t, fmt='%8.7f')
        np.savetxt(error_q_filename, error_q, fmt='%8.7f')

        # trajectory
        fig = plt.figure()
        real_pose = pred_translation - pose_m
        gt_pose = gt_translation - pose_m
        plt.scatter(gt_pose[:, 1], gt_pose[:, 0], s=1, c='black')
        plt.scatter(real_pose[:, 1], real_pose[:, 0], s=1, c='red')
        # plt.scatter(real_pose[:, 1], real_pose[:, 0], s=2, c=(5/255, 190/255, 251/255))
        # plt.plot(real_pose[:, 1], real_pose[:, 0], linewidth=1, color='red')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=10)
        image_filename = os.path.join(os.path.expanduser(cfg.TRAIN.exp_dir),
                                      '{:s}.png'.format('trajectory'))
        fig.savefig(image_filename, dpi=200, bbox_inches='tight')

        # translation_distribution
        fig = plt.figure()
        t_num = np.arange(len(error_t))
        plt.scatter(t_num, error_t, s=1, c='red')
        plt.xlabel('Data Num')
        plt.ylabel('Error (m)')
        image_filename = os.path.join(os.path.expanduser(cfg.TRAIN.exp_dir),
                                      '{:s}.png'.format('distribution_t'))
        fig.savefig(image_filename, dpi=200, bbox_inches='tight')

        # rotation_distribution
        fig = plt.figure()
        q_num = np.arange(len(error_q))
        plt.scatter(q_num, error_q, s=1, c='blue')
        plt.xlabel('Data Num')
        plt.ylabel('Error (degree)')
        image_filename = os.path.join(os.path.expanduser(cfg.TRAIN.exp_dir),
                                      '{:s}.png'.format('distribution_q'))
        fig.savefig(image_filename, dpi=200, bbox_inches='tight')

        # save error and trajectory
        error_t_filename = osp.join(cfg.TRAIN.exp_dir, 'error_t.txt')
        error_q_filename = osp.join(cfg.TRAIN.exp_dir, 'error_q.txt')
        pred_q_filename = osp.join(cfg.TRAIN.exp_dir, 'pred_q.txt')
        pred_t_filename = osp.join(cfg.TRAIN.exp_dir, 'pred_t.txt')
        gt_t_filename = osp.join(cfg.TRAIN.exp_dir, 'gt_t.txt')
        gt_q_filename = osp.join(cfg.TRAIN.exp_dir, 'gt_q.txt')
        np.savetxt(error_t_filename, error_t, fmt='%8.7f')
        np.savetxt(error_q_filename, error_q, fmt='%8.7f')
        np.savetxt(pred_t_filename, real_pose, fmt='%8.7f')
        np.savetxt(pred_q_filename, pred_rotation, fmt='%8.7f')
        np.savetxt(gt_q_filename, gt_rotation, fmt='%8.7f')
        np.savetxt(gt_t_filename, gt_pose, fmt='%8.7f')


if __name__ == '__main__':
    # conf = OmegaConf.load('config/xac.yaml')
    conf = OmegaConf.load('config/nclt.yaml')
    # conf = OmegaConf.load('config/qeoxford.yaml')
    LOG_FOUT = open(os.path.join(conf.TRAIN.exp_dir, 'log.txt'), 'w')
    LOG_FOUT.write(str(conf) + '\n')
    val_writer = SummaryWriter(os.path.join(conf.TRAIN.exp_dir, 'valid'))
    # 5 cpu core
    torch.set_num_threads(5)
    test(conf)