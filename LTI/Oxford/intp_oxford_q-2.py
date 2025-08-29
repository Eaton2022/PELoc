import h5py
import numpy as np
import os.path as osp
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# 旋转矩阵和位移向量的插值
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import os
import tqdm
from tqdm import tqdm
from scipy.spatial import KDTree

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline, interp1d



def compute_relative_transform(R1, t1, R2, t2):
    """
    计算从第一帧到第二帧的相对旋转矩阵和平移向量
    :param R1: 第一帧的旋转矩阵
    :param t1: 第一帧的平移向量
    :param R2: 第二帧的旋转矩阵
    :param t2: 第二帧的平移向量
    :return: 相对旋转矩阵和平移向量
    """
    # 计算相对旋转矩阵
    R_relative = np.dot(R2, R1.T)
    t1 = t1.reshape(3,-1)
    t_relative = np.dot(R_relative.T, (t2 - t1))
    return R_relative, t_relative.transpose()


def apply_relative_transform(cloud_1, R_relative, t_relative):
    """
    应用相对变换到第一帧点云
    :param cloud_1: 第一帧点云，形状为 (N, 3)
    :param R_relative: 相对旋转矩阵
    :param t_relative: 相对平移向量
    :return: 变换后的点云
    """
    return np.dot(cloud_1, R_relative.T) + t_relative


# 旋转矩阵和位移向量的插值
def interpolate_pose(R1, t1, R2, t2, t):

    # 创建四元数插值对象
    slerp = Slerp([0, 1], R.from_matrix([R1, R2]))

    # 使用SLERP插值
    interpolated_rotation = slerp(t)

    # 得到插值后的旋转矩阵
    interpolated_rotation_matrix = interpolated_rotation.as_matrix()

    # 对平移向量使用线性插值
    interpolated_translation = t1 + t * (t2 - t1)

    return interpolated_rotation_matrix, interpolated_translation.reshape(-1, 1)

if __name__ == '__main__':
    
    
    
    original_h5_path = osp.join(
    r"D:\文件\code_learn\3D-Detection-Tracking-Viewer-master\3D-Detection-Tracking-Viewer-master\h5",
    "velodyne_left_False.h5"
)
    expanded_h5_path = osp.join(
        r"D:\文件\code_learn\3D-Detection-Tracking-Viewer-master\3D-Detection-Tracking-Viewer-master\h5",
        "velodyne_left_False_init_t=0.5_bias=-2_3.4_q_6.9.h5"
    )
    
    
    
    with h5py.File(original_h5_path, 'r') as h5_file:
        ts = h5_file['valid_timestamps'][...]  # 加载时间戳
        ps = h5_file['poses'][...]             # 加载位姿矩阵
    
 
    # 检查原始数据的形状
    print(f"Original timestamps shape: {ts.shape}")          ### (37666,)
    print(f"Original poses shape: {ps.shape}")               ### (37666,12)

 
    point_cloud_pass = osp.join(r"D:\文件\Oxford\txt\SPVNAS_velodyne_left_plane_segmented_5")

    point_cloud_pass_3 = osp.join(r"D:\文件\Oxford\txt\SPVNAS_velodyne_left_plane_segmented_5_R_6.9") #### 3.4号增加旋转的影响
    
    file_pc = list(os.listdir(point_cloud_pass))
    
    len_pc = len(file_pc)
    t_values = [0.5]
    shift = -2
    shiftarr = np.array([0, shift, 0])
    # ps_expanded = np.zeros((37665,12))

    ps_expanded = np.zeros((37665,12))
    for i in tqdm(range(37665), desc="Saving intp pose", unit="file"):
        # R1 = ps[i][:9].reshape(3,3)
        R1 = ps[i,[0,1,2,4,5,6,8,9,10]].reshape(3,3)
        R2 = ps[i+1,[0,1,2,4,5,6,8,9,10]].reshape(3,3)
        t1 = ps[i,[3,7,11]]
        t2 = ps[i+1,[3,7,11]]
        # breakpoint()
        pc = np.fromfile(osp.join(r"D:\文件\Oxford\txt\SPVNAS_velodyne_left_plane_segmented_5", file_pc[i]),dtype = np.float32).reshape(-1,5)
        pc_cls = pc[:,-1].reshape(-1,1)
        pc_c = pc[:,:3]
        name_pc = file_pc[i].split('.')[0]
        for t in t_values:
            R_interp, t_interp = interpolate_pose(R1, t1, R2, t2, t)
            ### 增加旋转
            noise_angle = np.random.uniform(-2, 2, size=3)  # 2度以内的旋转噪声
            noise_rotation = R.from_euler('xyz', noise_angle, degrees=True).as_matrix()
            R_interp = noise_rotation @ R_interp
            # X = interpolate_pose(R1, t1, R2, t2, t)
            R_rel, T_rel = compute_relative_transform(R1, t1, R_interp, t_interp)
            R_in = R_interp.reshape(3,3)
            t_in = t_interp.reshape(1,3)
            random_offset = np.random.uniform(-0.3, 0.3, size=t_in.shape[0])  # 生成随机偏移量
            sinusoidal_offset = 0.1 * np.sin(t * np.pi * 2)  # 生成正弦扰动
            tmp = np.dot(R_in.reshape(3, 3), shiftarr)
            tmp[2] += random_offset
            t_in += tmp #random_offset

            t_in[:, 0] -= sinusoidal_offset
            new_pose = np.concatenate((R_in, t_in.reshape(3,1)),1).reshape(12,)
            ps_expanded[i] = new_pose #### 生成新的位姿

            a = R_interp.T @ R1 @ pc_c.T
            b = R_interp.T @ (0.5 * (t2 - t1).reshape(3, 1))
            transformed_points = (a - b).transpose()
            transformed_points[:, 1] -= shift + random_offset
            transformed_points[:, 0] -= sinusoidal_offset  # 同步正弦扰动
            # breakpoint()
            transformed_points = np.concatenate((transformed_points,pc_cls),axis = 1)
            save_path = osp.join(point_cloud_pass_3,f"{name_pc}_t=0.5_bais=-2.npy")
            np.save(save_path,transformed_points)
          
    print("done")    
    # breakpoint()
    np.save("ps_0.5_bia=-2.npy",ps_expanded)
    
    
    ps_expanded = ps_expanded[:37665]
            
    ## 修改时间戳和位姿    
    num_repeat = 1
    # 扩展时间戳，每个时间戳变成 时间戳_1, 时间戳_2, ...
    ts_expanded = np.array([f"{t}_t=0.5_bais=-2" for t in ts], dtype="S")   ####
    ts_expanded = ts_expanded[:37665]

    print(f"Expanded timestamps shape: {ts_expanded.shape}")
    print(f"Expanded poses shape: {ps_expanded.shape}")

    # 打印扩展后的时间戳数据
    print("Expanded timestamps:")
    print(ts_expanded)

    # 保存扩展后的数据到新的 H5 文件
    with h5py.File(expanded_h5_path, 'w') as new_h5_file:
        new_h5_file.create_dataset('valid_timestamps', data=ts_expanded)
        new_h5_file.create_dataset('poses', data=ps_expanded)

    print(f"Expanded data saved to {expanded_h5_path}")
