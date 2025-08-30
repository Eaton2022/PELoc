import re
import torch
import logging
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import torch.nn.functional as F
import numpy as np

_logger = logging.getLogger(__name__)


def Norm(norm_type, num_feats, bn_momentum=0.1, D=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')


class Conv(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 dimension=3):
        super(Conv, self).__init__()

        self.net = nn.Sequential(ME.MinkowskiConvolution(inplanes,
                                                         planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         dilation=dilation,
                                                         bias=bias,
                                                         dimension=dimension),)

    def forward(self, x):
        return self.net(x)


# class Encoder(ME.MinkowskiNetwork):
#     """
#     FCN encoder, used to extract features from the input point clouds.

#     The number of output channels is configurable, the default used in the paper is 512.
#     """

#     def __init__(self, out_channels, norm_type, D=3):
#         super(Encoder, self).__init__(D)

#         self.in_channels = 3
#         self.out_channels = out_channels
#         self.norm_type = norm_type
#         self.conv_planes = [32, 64, 128, 256, 256, 256, 256, 512, 512]

#         # in_channels, conv_planes, kernel_size, stride  dilation  bias
#         self.conv1 = Conv(self.in_channels, self.conv_planes[0], 3, 1)
#         self.norm1 = Norm(self.norm_type, self.conv_planes[0])
#         self.conv2 = Conv(self.conv_planes[0], self.conv_planes[1], 3, 2)
#         self.norm2 = Norm(self.norm_type, self.conv_planes[1])
#         self.conv3 = Conv(self.conv_planes[1], self.conv_planes[2], 3, 2)
#         self.norm3 = Norm(self.norm_type, self.conv_planes[2])
#         self.conv4 = Conv(self.conv_planes[2], self.conv_planes[3], 3, 2)
#         self.norm4 = Norm(self.norm_type, self.conv_planes[3])

#         self.res1_conv1 = Conv(self.conv_planes[3], self.conv_planes[4], 3, 1)
#         self.res1_norm1 = Norm(self.norm_type, self.conv_planes[4])
#         # 1
#         self.res1_conv2 = Conv(self.conv_planes[4], self.conv_planes[5], 1, 1)
#         self.res1_norm2 = Norm(self.norm_type, self.conv_planes[5])
#         self.res1_conv3 = Conv(self.conv_planes[5], self.conv_planes[6], 3, 1)
#         self.res1_norm3 = Norm(self.norm_type, self.conv_planes[6])

#         self.res2_conv1 = Conv(self.conv_planes[6], self.conv_planes[7], 3, 1)
#         self.res2_norm1 = Norm(self.norm_type, self.conv_planes[7])
#         # 2
#         self.res2_conv2 = Conv(self.conv_planes[7], self.conv_planes[8], 1, 1)
#         self.res2_norm2 = Norm(self.norm_type, self.conv_planes[8])
#         self.res2_conv3 = Conv(self.conv_planes[8], self.out_channels, 3, 1)
#         self.res2_norm3 = Norm(self.norm_type, self.out_channels)

#         self.res2_skip = Conv(self.conv_planes[6], self.out_channels, 1, 1)

#     def forward(self, x):
#         """
#         w/ BN
#         """
#         x = MEF.relu(self.norm1(self.conv1(x)))
#         x = MEF.relu(self.norm2(self.conv2(x)))
#         x = MEF.relu(self.norm3(self.conv3(x)))
#         res = MEF.relu(self.norm4(self.conv4(x)))

#         x = MEF.relu(self.res1_norm1(self.res1_conv1(res)))
#         x = MEF.relu(self.res1_norm2(self.res1_conv2(x)))
#         x._F = x.F.to(torch.float32)
#         x = MEF.relu(self.res1_norm3(self.res1_conv3(x)))

#         res = res + x

#         x = MEF.relu(self.res2_norm1(self.res2_conv1(res)))
#         x = MEF.relu(self.res2_norm2(self.res2_conv2(x)))
#         x._F = x.F.to(torch.float32)
#         x = MEF.relu(self.res2_norm3(self.res2_conv3(x)))

#         x = self.res2_skip(res) + x

#         return x


class WOBN_Encoder(ME.MinkowskiNetwork):
    """
    FCN encoder, used to extract features from the input point clouds.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self, out_channels, norm_type, D=3):
        super(WOBN_Encoder, self).__init__(D)

        self.in_channels = 3
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.conv_planes = [32, 64, 128, 256, 256, 256, 256, 512, 512]
        # self.conv_planes = [64, 128, 128, 256, 256, 256, 256, 512, 4096, 4096]

        # in_channels, conv_planes, kernel_size, stride  dilation  bias
        self.conv1 = Conv(self.in_channels, self.conv_planes[0], 3, 1, 1, True)        ### stride为2，尺寸减少 dilation：default     32
        self.conv2 = Conv(self.conv_planes[0], self.conv_planes[1], 3, 2, bias=True)   # * 64
        self.conv3 = Conv(self.conv_planes[1], self.conv_planes[2], 3, 2, bias=True)   # * 128
        self.conv4 = Conv(self.conv_planes[2], self.conv_planes[3], 3, 2, bias=True)   # * 256

        
        ## skip layers
        self.res1_conv1 = Conv(self.conv_planes[3], self.conv_planes[4], 3, 1, bias=True) # 256
        # 1
        self.res1_conv2 = Conv(self.conv_planes[4], self.conv_planes[5], 1, 1, bias=True) # 256  
        self.res1_conv3 = Conv(self.conv_planes[5], self.conv_planes[6], 3, 1, bias=True) # 256

        self.res2_conv1 = Conv(self.conv_planes[6], self.conv_planes[7], 3, 1, bias=True) # 512
        # 2
        self.res2_conv2 = Conv(self.conv_planes[7], self.conv_planes[8], 1, 1, bias=True) # 512
        self.res2_conv3 = Conv(self.conv_planes[8], self.out_channels, 3, 1, bias=True)   # 512

        self.res2_skip = Conv(self.conv_planes[6], self.out_channels, 1, 1, bias=True)    # 512

    def forward(self, x):
        """
        w/o BN
        """
  
        x = MEF.relu(self.conv1(x))
        x = MEF.relu(self.conv2(x))

        x = MEF.relu(self.conv3(x))

        res = MEF.relu(self.conv4(x))
    
        x = MEF.relu(self.res1_conv1(res))
        x = MEF.relu(self.res1_conv2(x))
        x._F = x.F.to(torch.float32)
        x = MEF.relu(self.res1_conv3(x))
        res = res + x   ### 1

        x = MEF.relu(self.res2_conv1(res))
        x = MEF.relu(self.res2_conv2(x))
        x._F = x.F.to(torch.float32)
        x = MEF.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x   ### 2

        return x




class Reg_Head(nn.Module):
    """
    nn.Linear版
    """
    def __init__(self, num_head_blocks, in_channels=512, mlp_ratio=1.0):
        super(Reg_Head, self).__init__()
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = in_channels  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels \
            else nn.Linear(self.in_channels, self.head_channels)

        block_channels = int(self.head_channels * mlp_ratio)
        self.res3_conv1 = nn.Linear(self.in_channels, self.head_channels)
        # self.res3_norm1 = nn.BatchNorm1d(self.head_channels)
        self.res3_conv2 = nn.Linear(self.head_channels, block_channels)
        # self.res3_norm2 = nn.BatchNorm1d(block_channels)
        self.res3_conv3 = nn.Linear(block_channels, self.head_channels)
        # self.res3_norm3 = nn.BatchNorm1d(self.head_channels)

        self.res_blocks = []
        self.norm_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Linear(self.head_channels, self.head_channels),
                # nn.BatchNorm1d(self.head_channels),
                nn.Linear(self.head_channels, block_channels),
                # nn.BatchNorm1d(block_channels),
                nn.Linear(block_channels, self.head_channels),
                # nn.BatchNorm1d(self.head_channels),
            ))

            super(Reg_Head, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            # super(Reg_Head, self).add_module(str(block) + 'b0', self.res_blocks[block][1])
            super(Reg_Head, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            # super(Reg_Head, self).add_module(str(block) + 'b1', self.res_blocks[block][3])
            super(Reg_Head, self).add_module(str(block) + 'c2', self.res_blocks[block][2])
            # super(Reg_Head, self).add_module(str(block) + 'b2', self.res_blocks[block][5])

        self.fc1 = nn.Linear(self.head_channels, self.head_channels)
        # self.fc1_norm = nn.BatchNorm1d(self.head_channels)
        self.fc2 = nn.Linear(self.head_channels, block_channels)
        # self.fc2_norm = nn.BatchNorm1d(block_channels)
        self.fc3 = nn.Linear(block_channels, 3)

        # glace
        # self.fcc = nn.Linear(block_channels, 100)
        # self.register_buffer("centers", mean.clone().detach().view(1, mean.shape[0], 3))
        # print(self.centers)

    def forward(self, res):
        """
        w/ BN
        """
   

        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        #
        res = self.head_skip(res) + x
        #
        for res_block in self.res_blocks:
   

            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))
        #
            res = res + x
      

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))



        sc = self.fc3(sc)

        return sc


class Regressor(ME.MinkowskiNetwork):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, num_head_blocks, num_encoder_features, level1_clusters=100, level2_clusters=25,
                 mlp_ratio=1.0, training=False, reg=False, D=3):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        """
        super(Regressor, self).__init__(D)

        self.feature_dim = num_encoder_features
        """
        ACE
        """
        self.encoder = WOBN_Encoder(out_channels=self.feature_dim, norm_type='BN')
        # self.cls_heads = Cls_Head(in_channels=self.feature_dim, level1_cluster=level1_clusters,
        #                           level2_cluster=level2_clusters, training=training)
        if reg:
            self.reg_heads = Reg_Head(num_head_blocks=num_head_blocks, in_channels=self.feature_dim, mlp_ratio=mlp_ratio)


    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        out = self.reg_heads(features)
        return out

    def get_scene_classification(self, features, lbl_1=None):
        # out_lbl_1, out_lbl_2 = self.cls_heads(features, lbl_1)
        out_lbl_1 = self.cls_heads(features, lbl_1)
        print(f"lbl_1: {lbl_1}, type: {type(lbl_1)}")
        # return out_lbl_1, out_lbl_2
        return out_lbl_1

    def forward(self, inputs):
        """
        Forward pass.
        """
        features = self.encoder(inputs)
        # breakpoint()
        out = self.get_scene_coordinates(features.F)
        # out = self.get_scene_classification(features)
        out = ME.SparseTensor(
            features=out,
            coordinates=features.C,
        )

        # return {'pred': out}
        return {'pred':out, 'f_512':features}
        