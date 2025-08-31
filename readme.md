## Unleashing the Power of Data Generation in One-Pass Outdoor LiDAR Localization  (ACM MM 2025)
------
📌 This repository contains the official implementation of our paper.

# PELoc
------
**PELoc** is the first LiDAR-based localization algorithm designed for **single-trajectory localization**.  
It achieves **near sub-meter accuracy** on most test trajectories of **QEOxford** and **NCLT** datasets.  
The core of PELoc lies in **pose enhancement**.

<p align="center">
  <img src="https://raw.githubusercontent.com/Eaton2022/PELoc/main/mm.png" width="700">
</p>

> 


## Visualization
------
<p align="center">
  <img src="https://raw.githubusercontent.com/Eaton2022/PELoc/main/2025-08-29_010302.png" width="500">
</p>

> 


## Experimental Results
------
- We provide experimental results for **Oxford**, **QEOxford**, and **NCLT** datasets in the `log` directory.  We provide QEOxford_pose_stats_PELoc.txt in the QEOxford folder, thus you can use our pretrained model to test performance in QEOxford dataset.
  
- Pretrained model weights are available on Baidu Netdisk:  

🔗 [Download Link](https://pan.baidu.com/s/1nwnnpqaF84gjtLF-Yc6MPw)  
🔑 Extraction Code: `nxte`


### LTI Usage
------
PELoc selects **2019-01-11-14-02-26** for Oxford/QEOxford  and **2012-02-18** for NCLT.  

We first use **LTI** to generate two additional trajectories offline.  You need to prepare both the **point cloud data** and the **pose files**,  then run the Python scripts in the `LTI` folder.

We provide two interpolation methods.  The LTI implementations for **NCLT** and **Oxford/QEOxford** have some differences,  but both significantly improve the accuracy of single-trajectory localization.

The **LTI_q** version achieves better accuracy improvement,  while the standard **LTI** version is more suitable for integration with **KP-CL**.  We will later upload our simulated point clouds to a cloud drive for download. You can also generate simulated point cloud trajectories by yourself.


**Note:** In our project, the trajectory point clouds and poses generated for Oxford/QEOxford are stored in the `2019-01-11-14-02-26` folder,  while those generated for NCLT are placed at the same level as `2012-02-18`.  Therefore, the way the point clouds and poses are loaded differs between these datasets.  Please pay attention to this when using our code.  For details, refer to: [lidarloc.py](https://github.com/Eaton2022/PELoc/blob/main/PELoc/datasets/lidarloc.py)


### Two Versions of PELoc
------
1. **PELoc-quick**  
   This version only requires **SSDA**, **LTI**, and optionally **RFT** for training.  It achieves high accuracy with much faster training speed.  You can use a batch size of **100** for this version.  

2. **PELoc-full**  
   This version additionally incorporates **KP-CL** during training.   It requires more training time, but achieves higher localization accuracy.

In the full code release, we provide the **PELoc-full** version. If you want to accelerate training, you may skip using **KP-CL**.

💡 Tip: If you are using the **PELoc-quick** version,  we recommend using the **LTI_q** version for generating simulated trajectories.   

Environment Setup
------ 
We provide a YAML file in the PELoc folder for environment configuration: [PELoc_env.yaml](https://github.com/Eaton2022/PELoc/blob/main/PELoc/SGLoc_env.yaml). Our experiments were conducted with CUDA 11.6, Python 3.8.20 on a single NVIDIA RTX 4090 GPU. In addition, our environment requires pytorch3d to be placed inside the PELoc directory, into pytorch3d folder, and then installed with:
~~~bash
pip install -e .
~~~
We provide a compatible version of pytorch3d via Baidu Netdisk:

🔗 [Download Link](https://pan.baidu.com/s/1HbPQRkg2WxpdEgeZ_znHZw)  
🔑 Extraction Code: `ytqa`

### Data Download
------ 
Please refer to this project for data preparation: [SGLoc](https://github.com/liw95/SGLoc).  

> **Note:** PELoc does **not** use semantic-segmented point clouds. Our experiments showed that segmentation does not improve performance. We use labeled point clouds because they were preprocessed before.

### Training and Testing
------ 
Set `x = 0` or `1` to select the GPU device.

```bash
#### Single-GPU Training

CUDA_VISIBLE_DEVICES=x accelerate launch --num_processes 1 --mixed_precision fp16 train.py

#### Single-GPU Testing
CUDA_VISIBLE_DEVICES=x test.py
```

### Acknowledgements
------ 
We sincerely thank the excellent open-source point cloud localization projects, including [SGLoc](https://github.com/liw95/SGLoc) and [LightLoc](https://github.com/liw95/LightLoc).



