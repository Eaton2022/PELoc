import h5py
import numpy as np
import os.path as osp
import MinkowskiEngine as ME
from torch.utils import data
from utils.augmentor import Augmentor, AugmentParams
from utils.pose_util import process_poses_nuscenes

BASE_DIR = osp.dirname(osp.abspath(__file__))


class KITTI(data.Dataset):
    def __init__(self, split='train'):
        # directories
        self.is_train = (split == 'train')

        data_path = 'ava16t/lw/Data'
        self.voxel_size = 0.4
        self.aug = True

        data_dir = osp.join(data_path, 'KITTI')
        if split == 'train':
            split_filename = osp.join(data_dir, 'train_split.txt')
        else:
            split_filename = osp.join(data_dir, 'valid_split.txt')
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]

        ps = {}
        ts = {}
        ss = {}
        pcs = []

        num_scene = 0

        for seq in seqs:
            seq_dir = osp.join(data_dir, 'sequences', seq)     # KITT/sequences/xx/
            # read the image timestamps
            h5_path = osp.join(seq_dir, 'poses_ws.h5')  # KITTI/sequences/xx/poses_ws.h5
            print(h5_path)

            if not osp.isfile(h5_path):
                # 生成真值文件
                # 1. 读入外参
                calib_lines = [line.rstrip('\n') for line in open(osp.join(data_dir, 'calib', seq, 'calib.txt'), 'r')]
                for calib_line in calib_lines:
                    if 'Tr' in calib_line:
                        velo_to_cam = np.zeros((4, 4))
                        velo2cam = calib_line.split(' ')[1:]
                        velo2cam = np.array(velo2cam, dtype='float').reshape(3, 4)
                        velo_to_cam[:3, :] = velo2cam
                        velo_to_cam[-1, -1] = 1
                # 2. 读入相机位姿真值
                pp = np.loadtxt(osp.join(data_dir, 'poses', seq + '.txt')).reshape(-1, 3, 4)   # KITT/poses/xx.txy
                p = np.zeros((len(pp), 4, 4))    # (n, 4, 4)
                p[:, :3, :] = pp
                p[:, -1, -1] = 1

                p = np.array([np.dot(pose, velo_to_cam) for pose in p])

                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))     #  (n, 12)
                ts[seq] = [i for i in range(len(p))]    #  [0, 1, ..., n]
                ss[seq] = [num_scene for i in range(len(p))]

                # write to h5 file
                print('write interpolate pose to ' + h5_path)
                h5_file = h5py.File(h5_path, 'w')
                h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
                h5_file.create_dataset('poses', data=ps[seq])
                h5_file.create_dataset('scenes', data =np.array(ss[seq], dtype=np.int64))

            else:
                # load h5 file, save pose interpolating time
                print("load " + seq + ' pose from ' + h5_path)
                h5_file = h5py.File(h5_path, 'r')
                ts[seq] = h5_file['valid_timestamps'][...]
                ps[seq] = h5_file['poses'][...]
                ss[seq] = h5_file['scenes'][...]

            num_scene +=1

            pcs.extend([osp.join(seq_dir, 'velodyne', str(t).zfill(6)+'.bin') for t in ts[seq]])

        vo_stats = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
        poses = np.concatenate([ps[seq] for seq in seqs], axis=0).reshape(-1, 3, 4)
        scenes_ids = np.concatenate([ss[seq] for seq in seqs], axis=0)
        scenes_ids = np.ones_like(scenes_ids)

        poses, rots = process_poses_nuscenes(poses_in=poses, scenes_ids=scenes_ids, num_scenes=self.num_scenes,
                                             align_R=vo_stats['R'], align_t=vo_stats['t'], align_s=vo_stats['s'])

        """
                self.pcs: 所有场景的文件名  
                self.poses: 文件对应的位姿
                scenes: 文件对应的场景
                """
        self.dataset_size = np.max(self.scenes_samples)

        # 构建字典，存储真值，包含pcs, poses, 以场景为索引
        # 初始化字典
        self.scene_data = {}
        for i, scene_idx in enumerate(scenes_ids):
            if scene_idx not in self.scene_data:
                self.scene_data[scene_idx] = {'pcs': [], 'poses': [], 'rots': []}
            self.scene_data[scene_idx]['pcs'].append(pcs[i])
            self.scene_data[scene_idx]['poses'].append(poses[i])
            self.scene_data[scene_idx]['rots'].append(rots[i])

        # scenes_pcs = []
        # mask = scenes_ids==(self.num_scenes-1)
        # self.poses = poses[mask]
        # self.rots = rots[mask]
        # for i in range(len(mask)):
        #     if mask[i]==True:
        #         scenes_pcs.append(pcs[i])
        # self.pcs = scenes_pcs

        if self.is_train:
            print("train data num:" + str(self.dataset_size))
        else:
            print("valid data num:" + str(self.dataset_size))

        augment_params = AugmentParams()
        augment_config = True

        self.aug_translation = 1
        self.aug_rotation = 10
        # Point cloud augmentations
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

        centers = np.zeros((25 * 100, 3))
        lbl = np.zeros((len(poses), 1))
        self.lbl_1 = lbl // 25
        self.lbl_2 = lbl % 25

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        i = 0
        scan_path = self.scene_data[i]['pcs'][idx]
        scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        scan = scan[:, :3]
        scan = np.ascontiguousarray(scan)
        pose = self.scene_data[i]['poses'][idx]
        rot = self.scene_data[i]['rots'][idx]
        scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)
        if self.is_train & self.aug:
            scan = self.augmentor.doAugmentation(scan)  # n, 5
        scan_gt_s8 = np.concatenate((scan, scan_gt), axis=1)
        coord, feat = ME.utils.sparse_quantize(
            coordinates=scan,
            features=scan_gt_s8,
            quantization_size=self.voxel_size)
        lbl_1 = self.lbl_1[idx]
        lbl_2 = self.lbl_2[idx]

        # 使得每次返回的数据为27个场景顺序叠加
        return (coord, feat, lbl_1, lbl_2, idx, pose)

