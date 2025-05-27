# RDIM

本项目基于可穿戴设备的帕金森病（PD）与鉴别性疾病（DD）在喝水活动中的分类任务，使用了公开数据集 PADS，核心代码基于 `KNNRDIP_pads_242-2-pads9.py` 实现。该研究已发表于 ICPADS 2023。

**论文题目：**  
*Modeling Parkinson’s Disease Aided Diagnosis with Multi-Instance Learning: An Effective Approach to Mitigate Label Noise*

**论文链接：**  
[https://ieeexplore.ieee.org/document/10476144](https://ieeexplore.ieee.org/document/10476144)

## 环境安装

请先安装 Conda 环境，然后在项目根目录下执行以下命令以根据 `is_env.yml` 创建运行环境：

```bash
conda env create -f is_env.yml
conda activate is_env
```

## 数据准备

本项目使用 PADS 数据集中 PD 和 DD 喝水活动的记录作为训练数据。请将数据放置到代码中预设的路径（可参考 `KNNRDIP_pads_242-2-pads9.py` 中的数据加载逻辑），确保数据路径正确。

## 运行代码

激活环境后，运行以下命令启动训练：

```

KNNRDIP_pads_242-2-pads9.py
```