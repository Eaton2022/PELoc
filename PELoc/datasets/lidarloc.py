import os
import h5py
import logging
import os.path as osp
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import Dataset
from sklearn.cluster import KMeans

from utils.pose_util import process_poses, calibrate_process_poses, build_se3_transform, process_poses_nuscenes
# For data augmentation
from utils.augmentor import Augmentor, AugmentParams
import concurrent.futures



_logger = logging.getLogger(__name__)
BASE_DIR = osp.dirname(osp.abspath(__file__))


# def compute_distances_chunk(xyz_chunk, centers):
#     """
#     计算一个xyz_chunk与centers之间的距离，并返回最小距离的索引。
#     """
#     # 计算xy坐标的欧几里得距离，假设我们只关心前两列
#     dist_mat = np.linalg.norm(xyz_chunk[:, np.newaxis, :2] - centers[np.newaxis, :, :2], axis=-1)
#     # 计算每行最小距离的索引
#     lbl_chunk = np.argmin(dist_mat, axis=1)
#     return lbl_chunk


# def parallel_kmeans(xyz, centers, n_jobs=8):
#     """
#     使用concurrent.futures并行计算距离矩阵，并返回最小距离的索引。
#     """
#     # 将xyz分成多个块
#     chunk_size = len(xyz) // n_jobs
#     xyz_chunks = [xyz[i:i + chunk_size] for i in range(0, len(xyz), chunk_size)]

#     # 使用concurrent.futures.ProcessPoolExecutor并行处理
#     with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
#         # 将每个块与centers一起传递到compute_distances_chunk函数
#         results = list(executor.map(compute_distances_chunk, xyz_chunks, [centers] * len(xyz_chunks)))

#     # 将所有结果合并为一个长数组
#     lbl = np.concatenate(results, axis=0)
#     return lbl


def poses_nuscenes(poses_in):
    poses_out = np.tile(np.eye(4), (len(poses_in), 1, 1))
    rot_out = np.zeros((len(poses_in), 3, 3))
    for i in range(len(poses_out)):
        R = poses_in[i, :9].reshape((3, 3))
        rot_out[i, :, :] = R
        poses_out[i, :3, :3] = R
        t = poses_in[i, 9:]
        poses_out[i, :3, 3] = t

    return poses_out

class LiDARLocDataset(Dataset):
    """LiDAR localization dataset.

    Access to point clouds, calibration and ground truth data given a dataset directory
    """

    def __init__(self,
                 root_dir,
                 train=True,
                 augment=False,
                 aug_rotation=10,
                 aug_translation=1,
                 voxel_size=0.25,
                 path = 3,
                 scene = 'QEOxford',
                 remove_points_ratio = 0):

        # Only support Oxford and NCLT
        self.root_dir = root_dir
        self.path = path
        print(remove_points_ratio,"!!!!!!!!")
        self.remove_points_ratio = remove_points_ratio
        self.train = train
        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_translation = aug_translation
        self.voxel_size = voxel_size
        self.k = 0
        # which dataset?
        # self.scene = osp.split(root_dir)[-1]
        self.scene = scene
        if self.scene == 'QEOxford':
            if self.train:
                # we use one original trajectory and two simulated trajetories for training
                seqs = ['2019-01-11-14-02-26-radar-oxford-10k', '-2_3.4_q','+2_3.4_q']    
            else:
                ## for test 
                # seqs = ['2019-01-15-13-06-37-radar-oxford-10k']
                # seqs = ['2019-01-17-13-26-39-radar-oxford-10k']
                seqs = ['2019-01-17-14-03-00-radar-oxford-10k']
                # seqs = ['2019-01-18-14-14-42-radar-oxford-10k']
        elif self.scene == 'Oxford':
            if self.train:
                # we use one original trajectory and two simulated trajetories for training
                seqs = ['2019-01-11-14-02-26-radar-oxford-10k', '-2_3.4_q_ox','+2_3.4_q_ox']    
            else:
                ## for test 
                # seqs = ['2019-01-15-13-06-37-radar-oxford-10k']
                # seqs = ['2019-01-17-13-26-39-radar-oxford-10k']
                seqs = ['2019-01-17-14-03-00-radar-oxford-10k']
                # seqs = ['2019-01-18-14-14-42-radar-oxford-10k']
        elif self.scene == 'NCLT':
            if self.train:
                seqs = ['2012-02-18', '2012-02-18-2-','2012-02-18-2+']
            else:
                seqs = ['2012-02-12']
                # seqs = ['2012-02-19']
                # seqs = ['2012-03-31']
                # seqs = ['2012-05-26']
        elif self.scene == 'XAC':
            if self.train:
                seqs = ['202311281', '202311282', '202311283', '202312067']
            else:
                # seqs = ['202312064']
                # seqs = ['202312065']
                seqs = ['202312066']
                # seqs = ['202312068']
        elif self.scene == 'nuScenes':
            if self.train:
                seqs = ['singapore_queenstown', 'singapore_onenorth', 'singapore_hollandvillage', 'boston_seaport']
            # extrinsic reading
            theta = -np.pi / 2  # -90 degrees in radians
            extrinsics = [0, 0, 0, 0, 0, theta]
            G_posesource_laser = build_se3_transform(extrinsics)
        elif self.scene == 'KITTI':
            if self.train:
                seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            else:
                seqs = ['00']
        else:
            raise RuntimeError('Only support Oxford/QEOxford and NCLT!')
        ps = {}
        ts = {}
        fs = {}
        vo_stats = {}
        self.pcs = []
        i = 0
        if (self.scene == 'QEOxford' or self.scene == 'Oxford'): seqs_root = seqs[0]
        for seq in seqs:
            if (self.scene == 'QEOxford' or self.scene == 'Oxford'):
                seq_dir = osp.join(self.root_dir, seqs_root)
            if self.scene == 'NCLT': 
                seq_dir = osp.join(self.root_dir, seq)
            i = i+1
            
            ## QEoxford
            if self.scene == 'QEOxford'and i == 1:
                h5_path = osp.join(self.root_dir, seq, 'velodyne_left_' + 'calibrateFalse.h5')   ####
            elif self.scene == 'QEOxford' and i == 2:
                h5_path = osp.join(self.root_dir, seqs_root, seq, 'velodyne_left_' + 'calibrateFalse_init_t=0.5_bias=-2_3.4_q.h5')   ####
            elif self.scene == 'QEOxford' and i == 3:
                h5_path = osp.join(self.root_dir, seqs_root, seq, 'velodyne_left_' + 'calibrateFalse_init_t=0.5_bias=+2_3.4_q.h5')     ####
            
            ## oxford   
            elif self.scene == 'Oxford'and i == 1:
                h5_path = osp.join(self.root_dir, seq, 'velodyne_left_' + 'False.h5')  ####
            elif self.scene == 'Oxford' and i == 2:
                h5_path = osp.join(self.root_dir, seqs_root, seq, 'velodyne_left_' + 'False_init_t=0.5_bias=-2_3.4_q.h5')   ####
            elif self.scene == 'Oxford' and i == 3:
                h5_path = osp.join(self.root_dir, seqs_root, seq, 'velodyne_left_' + 'False_init_t=0.5_bias=+2_3.4_q.h5')   ####
            
            
            ## NCLT
            elif self.scene == 'NCLT'and i == 1:
                # h5_path = osp.join(self.root_dir, seq, 'velodyne_left_' + 'calibrateFalse_init_t=0.5.h5')   ####
                h5_path = osp.join(self.root_dir, seq, 'velodyne_left_' + 'False.h5')  ####
            elif self.scene == 'NCLT' and i == 2:
                h5_path = osp.join(self.root_dir, seq, 'velodyne_left_' + 'False_init_t=0.5_bias=-2_3.4.h5')   ####
                # h5_path = osp.join(self.root_dir, seq, 'velodyne_left_' + 'calibrateFalse.h5')   ####
            elif self.scene == 'NCLT' and i == 3:
                h5_path = osp.join(self.root_dir, seq, 'velodyne_left_' + 'False_init_t=0.5_bias=+2_3.4.h5')   ####
            
            
            elif self.scene == 'nuScenes':
                h5_path = osp.join(self.root_dir, seq, 'poses_ws.h5')
            elif self.scene == 'KITTI':
                h5_path = osp.join(self.root_dir, 'sequences', seq, 'poses_ws.h5')
                seq_dir = osp.join(self.root_dir, 'sequences', seq)
            else:
                # print('I AM OXFORD')
                h5_path = osp.join(self.root_dir, seq, 'velodyne_left_' + 'False.h5')  ## oxford

            # load h5 file, save pose interpolating time  h5 ---> pose
            print("load " + seq + ' pose from ' + h5_path)
            h5_file = h5py.File(h5_path, 'r')
            if self.scene != 'nuScenes':
                ts[seq] = h5_file['valid_timestamps'][5:-5]
                ps[seq] = h5_file['poses'][5:-5]
                # ts[seq] = [t.decode('utf-8') for t in ts[seq]]
                print(self.scene,123456)
                if (self.scene == 'QEOxford' or self.scene == 'Oxford') and i == 1:
                    print("i am raw QEOxford/Oxford seg_pc1")   ### 增加一个判断句
                    self.pcs.extend(
                        ## you can use npy or bin file for training, and read files by np.load or np.fromfile
                        
                        # [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented', '{:d}.bin'.format(t)) for t in ts[seq]])   ### .bin
                        # [osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]]) ### .npy
                        [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented_npy', '{:d}.npy'.format(t)) for t in ts[seq]]) ### .npy
                    
                elif self.scene == 'QEOxford' and i == 2:
                    print("i am intp QEOxford seg_pc2")   ### 增加一个判断句
                    ts[seq] = [t.decode('utf-8') for t in ts[seq]]
                    self.pcs.extend(
                        [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented_npy_0.5_bias=-2_3.4_q', f'{t}.npy') for t in ts[seq]])   ### .npy   midu
                        # [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented', '{:d}.bin'.format(t)) for t in ts[seq]])   ### .bin
                        # [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented_npy', '{:d}.npy'.format(t)) for t in ts[seq]]) 
                    
                elif self.scene == 'QEOxford' and i == 3:
                    print("i am intp QEOxford seg_pc3")   ### 增加一个判断句
                    ts[seq] = [t.decode('utf-8') for t in ts[seq]]
                    self.pcs.extend(
                        [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented_npy_0.5_bias=+2_3.4_q', f'{t}.npy') for t in ts[seq]])  ### .npy  midu
                
                ## Oxford
                elif self.scene == 'Oxford' and i == 2:
                    print("i am intp Oxford seg_pc2")   ### 增加一个判断句
                    ts[seq] = [t.decode('utf-8') for t in ts[seq]]
                    self.pcs.extend(
                        [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented_npy_0.5_bias_oxford=-2_3.4_q', f'{t}.npy') for t in ts[seq]])
                    
                elif self.scene == 'Oxford' and i == 3:
                    print("i am intp Oxford seg_pc3")   ### 增加一个判断句
                    ts[seq] = [t.decode('utf-8') for t in ts[seq]]
                    self.pcs.extend(
                        [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented_npy_0.5_bias_oxford=+2_3.4_q', f'{t}.npy') for t in ts[seq]])
                
                
                ## NCLT
                elif self.scene == 'NCLT' and i == 1:
                    print("i am raw NCLT seg_pc1")   ### 增加一个判断句
                    #### self.pcs 可以增加centers_with_node
                    self.pcs.extend(
                        # # [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented_npy_0.5', f'{t}.npy') for t in ts[seq]])   ### .bin
                        [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented', '{:d}.bin'.format(t)) for t in ts[seq]])   ### .bin
                        # [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented_npy', '{:d}.npy'.format(t)) for t in ts[seq]]) ### .npy
                    
                elif self.scene == 'NCLT' and i == 2:
                    print("i am intp NCLT seg_pc2")   ### 增加一个判断句
                    ts[seq] = [t.decode('utf-8') for t in ts[seq]]
                    self.pcs.extend(
                        [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented-2', f'{t}.bin') for t in ts[seq]])   ### .npy   midu
                        # [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented-2', '{:d}.bin'.format(t)) for t in ts[seq]])   ### .bin
                        # [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented_npy', '{:d}.npy'.format(t)) for t in ts[seq]]) 
                    
                elif self.scene == 'NCLT' and i == 3:
                    print("i am intp NCLT seg_pc3")   ### 增加一个判断句
                    ts[seq] = [t.decode('utf-8') for t in ts[seq]]
                    self.pcs.extend(
                        [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented+2', f'{t}.bin') for t in ts[seq]])   ### .npy   midu
                
                
                # elif self.scene != 'KITTI' and i == 4:
                #     print("i am intp seg_pc4")   ### 增加一个判断句
                #     ts[seq] = [t.decode('utf-8') for t in ts[seq]]
                #     self.pcs.extend(
                #         [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented_npy_0.5_bias=-4+j', f'{t}.npy') for t in ts[seq]])   ### .npy
                
                
                
                # elif self.scene != 'KITTI' and i == 5:
                #     print("i am intp seg_pc5")   ### 增加一个判断句
                #     ts[seq] = [t.decode('utf-8') for t in ts[seq]]
                #     self.pcs.extend(
                #         [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented_npy_0.5_bias=+4+j', f'{t}.npy') for t in ts[seq]])   ### .npy
                else:
                    self.pcs.extend([osp.join(seq_dir, 'velodyne', str(t).zfill(6) + '.bin') for t in ts[seq]])
            else:
                file_names = h5_file['file_names'][...][5:-5]
                fs[seq] = [name.decode('utf-8') for name in file_names]
                p = h5_file['poses'][5:-5]
                pose = poses_nuscenes(p)
                # 与外参相乘
                p = np.asarray([np.dot(p, G_posesource_laser) for p in pose])  # (n, 4, 4)
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))  # (n, 12)
                self.pcs.extend([osp.join('/3D/lw/nuScenes', t) for t in fs[seq]])

            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}


        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))

        kmeans_pose_stats_filename = osp.join(self.root_dir, self.scene + '_cls_pose_stats.txt')
        mean_pose_stats_filename = osp.join(self.root_dir, self.scene + '_pose_stats.txt')
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))

        if self.train:
            if self.scene == 'QEOxford':
                # 计算轨迹的中值
                self.mean_t = np.mean(poses[:, 9:], axis=0)  # (3,)
                # 存储
                np.savetxt(mean_pose_stats_filename, self.mean_t, fmt='%8.7f')
            else:
                # 计算轨迹的中值
                self.mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
                # 存储
                np.savetxt(mean_pose_stats_filename, self.mean_t, fmt='%8.7f')
        else:
            self.mean_t = np.loadtxt(mean_pose_stats_filename)
            # breakpoint()

        for seq in seqs:
            # breakpoint()
            if self.scene == 'QEOxford':         
                print("i am QEOxford")       
                pss, rotation = calibrate_process_poses(poses_in=ps[seq], mean_t=self.mean_t, align_R=vo_stats[seq]['R'],
                                                        align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s'])
            else:
                print("i am oxford or NCLT")
                pss, rotation = process_poses(poses_in=ps[seq], mean_t=self.mean_t, align_R=vo_stats[seq]['R'],
                                              align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
            self.rots = np.vstack((self.rots, rotation))

        xyz = self.poses[:, :3]

        # data augment
        augment_params = AugmentParams()
        if self.augment:
            augment_params.setTranslationParams(
                p_transx=0.5, trans_xmin=-1 * self.aug_translation, trans_xmax=self.aug_translation,
                p_transy=0.5, trans_ymin=-1 * self.aug_translation, trans_ymax=self.aug_translation,
                p_transz=0, trans_zmin=-1 * self.aug_translation, trans_zmax=self.aug_translation)
            augment_params.setRotationParams(
                p_rot_roll=0, rot_rollmin=-1 * self.aug_rotation, rot_rollmax=self.aug_rotation,
                p_rot_pitch=0, rot_pitchmin=-1 * self.aug_rotation, rot_pitchmax=self.aug_rotation,
                p_rot_yaw=0.5, rot_yawmin=-1 * self.aug_rotation, rot_yawmax=self.aug_rotation)
            self.augmentor = Augmentor(augment_params)
        else:
            self.augmentor = None

        if self.train:
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))
            
    ## RFT
    def remove_continuous_frames(self,idx_list,num_remove_ratio):
        """
        随机去掉一段连续的帧
        :param idx_list: 轨迹中的帧索引
        :param num_remove_ratio: 需要去掉的比例 (默认 1/20)  5%的连续帧, 让其失去对轨迹的依赖
        :return: 处理后的帧索引
        """
        path_num = self.path
        # num_remove_ratio = self.remove_points_ratio
        num_frames = len(idx_list)//path_num    ####37656
        num_remove = max(1, int(num_frames * num_remove_ratio))  # 每段去掉多少帧
        total_new_list = list()      
        for i in range(path_num):
            # 获取该轨迹的索引
            start_idx_in_list = i * num_frames
            end_idx_in_list = (i + 1) * num_frames
            new_idx_list = idx_list[start_idx_in_list:end_idx_in_list]  # 取出当前轨迹的帧索引

            # 随机选择一个起点，去掉 `num_remove` 帧
            start_idx = np.random.randint(0, num_frames - num_remove + 1)
            # print(start_idx)
            new_idx_list = new_idx_list[:start_idx] + new_idx_list[start_idx + num_remove:]
            total_new_list += new_idx_list 
        return total_new_list

               
    def __len__(self):
        index_list = list(range(len(self.poses)))
        self.index_list = self.remove_continuous_frames(index_list,self.remove_points_ratio)   ### RFT Training Strategy
        return len(self.index_list)    
        # return len(self.poses)

    def __getitem__(self, idx):
        if type(idx) == list:
            scenes = [0]
            num = 0
            for i in idx:
                scan_path = self.pcs[i]
                if self.scene != 'KITTI':
                    ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 5)[:, :3]
                else:
                    ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)[:, :3]
                if self.scene != 'XAC' and self.scene != 'nuScenes' and self.scene != 'KITTI':
                    ptcld[:, 2] = -1 * ptcld[:, 2]

                scan = ptcld

                scan = np.ascontiguousarray(scan)

                lbl_1 = self.lbl_1[i]
                lbl_2 = self.lbl_2[i]

                pose = self.poses[i]
                rot = self.rots[i]
                scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)   ### 镜像翻转需要改变真值  有可能还需要再改
                
                
                ### 增强
                if self.train & self.augment:
                    scan = self.augmentor.doAugmentation(scan)  # n, 5
         
                
             
                scan_gt_s8 = np.concatenate((scan, scan_gt), axis=1)
       
                
                
                
                ### 坐标
                coord, feat = ME.utils.sparse_quantize(
                    coordinates=scan,
                    features=scan_gt_s8,
                    quantization_size=self.voxel_size)

  
                        
                        
                if num == 0:
                    coords = coord
                    feats = feat
                    poses = pose.reshape(1, 6)
                    scenes.append(len(coord))
                else:
                    coords = np.concatenate((coords, coord))
                    feats = np.concatenate((feats, feat))
                    poses = np.concatenate((poses, pose.reshape(1, 6)))
                    scenes.append(scenes[num] + len(coord))
                num += 1

            return (coords, feats, poses, scenes)

        else:

            scan_path = self.pcs[idx]
            # print("-------")
            if self.scene != 'KITTI':
                # ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(4, -1).transpose()[:,:3]
                # ptcld[:, 2] = -1 * ptcld[:, 2]
                # ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)[:,:3]   ##### .bin 2.24
                # ptcld[:, 2] = -1 * ptcld[:, 2]
                # ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 5)[:, :3]    ##### .bin 2.24 1814
                if self.scene in ['QEOxford', 'Oxford']:
                    if scan_path.endswith(".npy"):
                        ptcld = np.load(scan_path)[:, :3]
                    elif scan_path.endswith(".bin"):
                        ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)[:, :3]
                    else:
                        raise ValueError(f"Unsupported point cloud format: {scan_path}")
                ## NCLT
                else:
                    ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 5)[:, :3]   # For NCLT
            else:
                print(scan_path)
                ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)[:, :3]
                # ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(4, -1).transpose()[:,:3]

            if self.scene != 'XAC' and self.scene != 'nuScenes' and self.scene != 'KITTI':
                ptcld[:, 2] = -1 * ptcld[:, 2]   ## REMOVE IT IF 1814

            scan = ptcld

            scan = np.ascontiguousarray(scan)

            pose = self.poses[idx]  # (6,)
            rot = self.rots[idx]  # [3, 3]
            # print(scan.shape)
            scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)

            if self.train & self.augment:
                scan = self.augmentor.SSDA(scan)  # n, 5
    
            scan_gt_s8 = np.concatenate((scan, scan_gt), axis=1)
            

            coords, feats = ME.utils.sparse_quantize(                   #### coords: torch.Size: N,3
                coordinates=scan,
                features=scan_gt_s8,
                quantization_size=self.voxel_size)
            
        return (coords, feats, idx, pose)

