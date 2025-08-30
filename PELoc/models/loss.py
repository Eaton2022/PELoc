import torch
from torch import nn
import torch.nn.functional as F



class KPCL(nn.Module):
    def __init__(self, temperature=0.15, projection_dim=128):  ### 0.15--->0.1    arg 2
        super(KPCL, self).__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(512, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        ).cuda()

    def augment_features(self, features):
        random_noise = torch.rand_like(features).to(features.device)
        normalized_noise = F.normalize(random_noise, dim=-1, eps=1e-8)
        features += torch.sign(features) * normalized_noise * 0.1          
        return features

    def forward(self, features):
        device = features.device
        self.projection_head = self.projection_head.to(device)
        
        if features.shape[0] > 17000:
            features = features[:17000,:]

        
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("Warning: NaN or Inf detected in features after flipping!")

        
        # 数据增强
        view1 = self.augment_features(features)  # (N, D)
        view2 = self.augment_features(features)  # (N, D)
        

        # 投影头
        features = self.projection_head(features)  # (N, D)
        view1 = self.projection_head(view1)  # (N, D)
        view2 = self.projection_head(view2)  # (N, D)

        # 归一化
        features = F.normalize(features, dim=1,eps=1e-6)  # (N, D)
        view1 = F.normalize(view1, dim=1,eps=1e-6)  # (N, D)
        view2 = F.normalize(view2, dim=1,eps=1e-6)  # (N, D)

        # 计算相似度矩阵
        similarity_matrix_view1 = torch.mm(features, view1.T) / self.temperature  # (N, N)
        similarity_matrix_view2 = torch.mm(features, view2.T) / self.temperature  # (N, N)
        similarity_matrix = torch.cat([similarity_matrix_view1, similarity_matrix_view2], dim=1)  # (N, 2N)

        # similarity_matrix_view = torch.mm(view1, view2.T) / self.temperature  # (N, N)
        
        
        # 正样本对（对角线元素）
        positive_scores = torch.diag(similarity_matrix[:features.size(0), :features.size(0)])


        # 计算 InfoNCE 损失
        loss = -positive_scores + torch.logsumexp(similarity_matrix, dim=1)  # (N,)
        return loss.mean()

class CriterionCoordinate(nn.Module):
    def __init__(self):
        super(CriterionCoordinate, self).__init__()
        self.KP_CL = KPCL()
        
    def forward(self, pred_point, gt_point,features,temperature = 0.15):
        ### loc loss
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        loss_map = torch.mean(loss_map)

        # KP CL loss
        loss_cl = self.KP_CL(features)
        # print(loss_cl*2)
        loss_map = loss_map +2*loss_cl

        return loss_map

# ########## SGLoc Loss ##########
# L1
class CriterionCoordinate_InfoBatch(nn.Module):
    def __init__(self, first_prune_epoch=None, second_prune_epoch=None, windows=None):
        super(CriterionCoordinate_InfoBatch, self).__init__()
        self.first_prune_epoch = first_prune_epoch
        self.second_prune_epoch = second_prune_epoch
        self.windows = windows

    def forward(self, pred_point, gt_point, batch_size=None, epoch_nums=None, batch_nums=None, idx=None, values=None):
        """
        pred_point: 预测值；
        gt_point: 真值；
        batch_size: 当前的batch_size数量
        epoch_nums: 当前epoch数；
        batch_nums: 体素的坐标列；
        idx: 当前的样本索引；
        values: [len(dataset), windows]矩阵，记录误差
        """
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        if self.first_prune_epoch <= epoch_nums < (self.first_prune_epoch + self.windows):
            loss_lw = loss_map.detach().clone()
            current_epoch = epoch_nums - self.first_prune_epoch
            # print(batch_nums.shape)
            # print(loss_map.shape)
            for i in range(batch_size):
                # 取出当前序列中的第N个batch数据
                mask = batch_nums == i
                values[idx[i], current_epoch] = torch.median(loss_lw[mask])   # 将其求中值，以忽略噪点影响
        elif self.second_prune_epoch <= epoch_nums < (self.second_prune_epoch + self.windows):
            loss_lw = loss_map.detach().clone()
            current_epoch = epoch_nums - self.second_prune_epoch
            for i in range(batch_size):
                # 取出当前序列中的第N个batch数据
                mask = batch_nums == i
                values[idx[i], current_epoch] = torch.median(loss_lw[mask])  # 将其求中值，以忽略噪点影响

        loss_map = torch.mean(loss_map)

        return loss_map, values


class Cls_Criterion(nn.Module):
    def __init__(self):
        super(Cls_Criterion, self).__init__()
        self.loc_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, pred_loc, gt_loc):
        loss = self.loc_loss(pred_loc, gt_loc.long())

        return loss
