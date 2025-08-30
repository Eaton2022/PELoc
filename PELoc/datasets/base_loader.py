import logging
import torch
import numpy as np
import MinkowskiEngine as ME


class CollationFunctionFactory:
    def __init__(self, collation_type='default'):
        if collation_type == 'default':
            self.collation_fn = self.collate_default
        elif collation_type == 'collate_pair_cls':
            self.collation_fn = self.collate_pair_cls_fn
        elif collation_type == 'collate_pair_cls_ace':
            self.collation_fn = self.collate_pair_cls_ace_fn


    def __call__(self, list_data):
        return self.collation_fn(list_data)

    def collate_default(self, list_data):
        return list_data

    def collate_pair_cls_fn(self, list_data):
        # list_data = [list_data]
        N = len(list_data)
        # print("N is %d"%N)
        list_data = [data for data in list_data if data is not None]
        if N != len(list_data):
            logging.info(f"Retain {len(list_data)} from {N} data.")

        if len(list_data) == 0:
            raise ValueError('No data in the batch')

        # coords, feats, label, scene = list(zip(*list_data))
        # breakpoint()
        # coords, feats, lbl_1, lbl_2, idx, pose = list(zip(*list_data))
        coords, feats,  idx, pose = list(zip(*list_data))
        # coords, coords2 = zip(*coords_list)
        # feats, feats2 = zip(*feats_list)
        # coords, feats, lbl_1, lbl_2, idx, pose, coords_ds = list(zip(*list_data))
        # lbl_1_batch = torch.from_numpy(np.stack(lbl_1)).to(torch.int64)
        # lbl_2_batch = torch.from_numpy(np.stack(lbl_2)).to(torch.int64)
        idx_batch = torch.from_numpy(np.stack(idx)).to(torch.int64)
        pose_batch = torch.from_numpy(np.stack(pose)).float()
        coords_batch = ME.utils.batched_coordinates(coords)
        # coords_batch_2 = ME.utils.batched_coordinates(coords2)
        # coords_ds_batch = torch.from_numpy(np.stack(coords_ds)).float()    ### (b, 4096, 3 )
        feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
        # feats_batch_2 = torch.from_numpy(np.concatenate(feats2, 0)).float()

        return {
            'sinput_C': coords_batch,
            # 'sinput_C_2': coords_batch_2,
            'sinput_F': feats_batch,
            # 'sinput_F_2': feats_batch_2,
            # 'lbl_1': lbl_1_batch,
            # 'lbl_2': lbl_2_batch,
            'idx': idx_batch,
            'pose': pose_batch,
            # 'sinput_C_ds': coords_ds_batch
        }

    def collate_pair_cls_ace_fn(self, list_data):
        list_data = [list_data]
        N = len(list_data)
        # print("N is %d"%N)
        list_data = [data for data in list_data if data is not None]
        if N != len(list_data):
            logging.info(f"Retain {len(list_data)} from {N} data.")

        if len(list_data) == 0:
            raise ValueError('No data in the batch')

        coords, feats, lbl_1, lbl_2, pose, scene = list(zip(*list_data))
        lbl_1_batch = torch.from_numpy(np.stack(lbl_1)).to(torch.int64)
        lbl_2_batch = torch.from_numpy(np.stack(lbl_2)).to(torch.int64)
        pose_batch = torch.from_numpy(np.stack(pose)).float()
        self.num_scenes = pose_batch.size(1)
        coords_batch = self.batched_seq_coordinates(coords, scene)
        feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()

        return {
            'sinput_C': coords_batch,
            'sinput_F': feats_batch,
            'lbl_1': lbl_1_batch,
            'lbl_2': lbl_2_batch,
            'pose': pose_batch
        }

    def batched_seq_coordinates(self, coords, scenes, dtype=torch.int32, device=None):
        D = np.unique(np.array([cs.shape[1] for cs in coords]))
        D = D[0]
        if device is None:
            if isinstance(coords, torch.Tensor):
                device = coords[0].device
            else:
                device = "cpu"
        N = np.array([len(cs) for cs in coords]).sum()
        bcoords = torch.zeros((N, D + 1), dtype=dtype, device=device)  # uninitialized
        s = 0
        scenes = list(scenes)
        for i in range(len(scenes)):
            scenes[i] = list(scenes[i])
            if i > 0:
                scenes[i] = [x + scenes[i-1][-1] for x in scenes[i]]
        for b, cs in enumerate(coords):
            if dtype == torch.int32:
                if isinstance(cs, np.ndarray):
                    cs = torch.from_numpy(np.floor(cs))
                elif not (
                        isinstance(cs, torch.IntTensor) or isinstance(cs, torch.LongTensor)
                ):
                    cs = cs.floor()

                cs = cs.int()
            else:
                if isinstance(cs, np.ndarray):
                    cs = torch.from_numpy(cs)
            if b > 0:
                for i in range(self.num_scenes):
                    bcoords[scenes[b][i]: scenes[b][i + 1], 1:] = cs[scenes[b][i] - scenes[b - 1][-1]:
                                                                     scenes[b][i + 1] - scenes[b - 1][-1]]
                    bcoords[scenes[b][i]: scenes[b][i + 1], 0] = s
                    s += 1
            else:
                for i in range(self.num_scenes):
                    bcoords[scenes[b][i]: scenes[b][i + 1], 1:] = cs[scenes[b][i]: scenes[b][i + 1]]
                    bcoords[scenes[b][i]: scenes[b][i + 1], 0] = s
                    s += 1
            # print("*"*20)
        return bcoords