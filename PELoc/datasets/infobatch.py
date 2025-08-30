"""
这份代码用于实现1. 达到一定epoch，通过中值损失裁剪数据；2. 达到一定epoch，恢复全部数据
"""
import math
import torch
import random
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import Dataset


class InfoBatch(Dataset):
    """
    Args:
        dataset: Dataset used for training.
        num_epochs (int): The number of epochs for pruning.
        prune_ratio (float, optional): The proportion of samples being pruned during training.
        delta (float, optional): The first delta * num_epochs the pruning process is conducted. It should be close to 1. Defaults to 0.875.
    """

    def __init__(self, dataset: Dataset, num_epochs: int,
                 prune_ratio: float = 0.25, delta_start: float = 0.25, windows_radio: float = 0.1,
                 delta_stop: float = 0.85):
        self.dataset = dataset
        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.prune_ratio = prune_ratio
        self.num_epochs = num_epochs
        self.delta_start = delta_start
        self.delta_stop = delta_stop
        self.windows = windows_radio    # 0.1
        self.keep_windows_ratio = ((delta_stop - delta_start) - 2 * windows_radio) / 2      # 0.2
        # self.scores stores the loss value of each sample. Note that smaller value indicates the sample is better learned by the network.
        self.weights = torch.ones(len(self.dataset), 1)
        self.select_idx = torch.zeros(len(dataset), dtype=torch.bool)
        self.num_pruned_samples = 0
        self.cur_batch_index = None

    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        self.cur_batch_index = cur_batch_indices

    def update(self, loss, idx, batch_num):
        """
        缩放损失
        """
        device = loss.device
        weights = self.weights.to(device)[idx][batch_num.long()]
        # print(weights.shape)
        # print(loss.shape)
        loss.mul_(weights)

        return loss.mean()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        scan_path = self.dataset.pcs[idx]
        ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        ptcld[:, 2] = -1 * ptcld[:, 2]

        scan = ptcld

        scan = np.ascontiguousarray(scan)

        lbl_1 = self.dataset.lbl_1[idx]
        lbl_2 = self.dataset.lbl_2[idx]
        pose = self.dataset.poses[idx]  # (6,)
        rot = self.dataset.rots[idx]  # [3, 3]
        scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)

        if self.dataset.train & self.dataset.augment:
            scan = self.dataset.augmentor.doAugmentation(scan)  # n, 5

        scan_gt_s8 = np.concatenate((scan, scan_gt), axis=1)

        coords, feats = ME.utils.sparse_quantize(
            coordinates=scan,
            features=scan_gt_s8,
            quantization_size=self.dataset.voxel_size)

        return (coords, feats, lbl_1, lbl_2, idx, pose)

    # 这个函数是在应用第一次裁剪前，更新select_idx，选择学习的不好的位置
    def cal_std(self, values, labels, label_num):
        """
        这里进来的应该是N个epoch样本的 <中值误差>, 计算方差
        values 应该有len(dataset)行， M列，取决于多少个epoch计算得到
        """
        # 1. 计算每行的方差
        variances = values.var(dim=1)

        # 2. 对每个类别单独计算
        for i in range(label_num):
            # 获取该类别的样本索引
            label = i
            label_indices = torch.where(labels == label)[0]
            # # 按照方差从大到小排序，并选择保留前keep_ratio百分比样本
            label_variances = variances[label_indices]
            num_top_samples = int(self.keep_ratio * len(label_variances))
            # 获取前 prune_ratio百分比 方差最大的样本的索引
            _, top_indices = torch.topk(label_variances, num_top_samples)
            selected_indices = label_indices[top_indices]
            # 将这些样本的位置在掩码中设置为 True
            self.select_idx[selected_indices] = True

        remain_idx = torch.where(self.select_idx)[0]
        self.weights[remain_idx] = self.weights[remain_idx] * (1 / self.keep_ratio)
        # 返回选择了多少样本，用于下一次裁剪
        return len(remain_idx), remain_idx

    # 这个函数应用第二次裁剪，更新select_idx
    def sec_cal_std(self, values, labels, selected_indices, label_num):
        """
        第二次裁剪
        这里进来的应该是N个epoch样本的 <中值误差>, 计算方差
        values 应该有len(sub_dataset)行， M列，取决于多少个epoch计算得到
        """
        # 1. 计算每行的方差
        values = values[selected_indices]
        print(values)
        labels = labels[selected_indices]
        print(labels)
        variances = values.var(dim=1)
        self.select_idx[:] = False

        # 2. 对每个类别单独计算
        for i in range(label_num):
            # 获取该样本的索引
            label = i
            label_indices = torch.where(labels == label)[0]
            # # 按照方差从大到小排序，并选择保留前keep_ratio百分比样本
            label_variances = variances[label_indices]
            num_top_samples = int(self.keep_ratio * len(label_variances))
            # # 获取前 prune_ratio百分比 方差最大的样本的索引
            _, top_indices = torch.topk(label_variances, num_top_samples)
            second_selected_indices = label_indices[top_indices]
            self.select_idx[second_selected_indices] = True

        remain_idx = torch.where(self.select_idx)[0]
        self.weights[remain_idx] = self.weights[remain_idx] * (1 / self.keep_ratio)

        return torch.sum(self.select_idx == True)

    def prune(self):
        # Prune samples that are well learned, rebalance the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance
        remained_mask = self.select_idx.numpy()
        remained_indices = np.where(remained_mask)[0].tolist()
        # well_learned_indices = np.where(~remained_mask)[0]
        # selected_indices = np.random.choice(well_learned_indices, int(
        #     self.keep_ratio * len(well_learned_indices)), replace=False)
        # if len(selected_indices) > 0:
        #     remained_indices.extend(selected_indices)
        # self.num_pruned_samples += len(self.dataset) - len(remained_indices)
        np.random.shuffle(remained_indices)

        return remained_indices

    @property
    def sampler(self):
        sampler = IBSampler(self)
        return sampler

    def no_prune(self):
        samples_indices = list(range(len(self)))
        np.random.shuffle(samples_indices)
        return samples_indices

    def reset_weights(self):
        self.weights[:] = 1

    @property
    def first_prune(self):
        # 向下取整
        return math.floor(self.num_epochs * self.delta_start)

    @property
    def stop_prune(self):
        # 向下取整
        return math.floor(self.num_epochs * self.delta_stop)

    @property
    def second_prune(self):
        # 向下取整
        # ceil(25 * (0.25 + 0.1 + 0.2) = 13.75) = 14
        return math.floor(self.num_epochs * (self.delta_start + self.windows + self.keep_windows_ratio))

    @property
    def win_std(self):
        # 向上取整
        return math.ceil(self.num_epochs * self.windows)


class IBSampler(object):
    def __init__(self, dataset: InfoBatch):
        self.dataset = dataset
        self.first_prune = dataset.first_prune
        self.stop_prune = dataset.stop_prune
        self.windows = dataset.win_std
        self.iterations = -1
        self.sample_indices = None
        self.iter_obj = None
        self.reset()

    def reset(self):
        # np.random.seed(self.iterations)
        print("iterations: %d" % self.iterations)
        if self.iterations >= self.stop_prune or self.iterations < (self.first_prune + self.windows):
            print("no pruning")
            if self.iterations == self.stop_prune:
                self.dataset.reset_weights()
            # print('we are going to stop prune, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            self.sample_indices = self.dataset.no_prune()
        else:
            print("pruning")
            # print('we are going to continue pruning, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            self.sample_indices = self.dataset.prune()
            print(len(self.sample_indices))
        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1

    def __next__(self):
        nxt = next(self.iter_obj)
        return nxt

    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        self.reset()
        return self
