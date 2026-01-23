# WD Projects: 3D 场景重建与生成工具箱

这是一个集成了多种前沿 3D 视觉算法的综合工程，涵盖了从图像去畸变、多视角几何重建、BEV（鸟瞰图）生成到新视角合成（View Synthesis）的全流程工具链。

## 📋 功能模块

本项目主要包含以下核心模块：

### 1. 图像去畸变 (Undistortion)
提供基于学习的相机标定与去畸变功能，支持野外（In-the-wild）图像的预处理。
- **支持算法**:
  - `geocalib`: 基于 GeoCalib 的自动标定。
  - `anycalib`: 通用标定工具。

### 2. 3D 场景重建 (Reconstruction)
集成了多种 SOTA 重建算法，用于从图像序列恢复稀疏点云、相机位姿和深度图。
- **支持算法**:
  - `mast3r`: 基于 MASt3R 的稠密匹配与重建。
  - `da3`: 集成 Depth Anything V3 的重建流程。
  - `hunyuanworld`: HunyuanWorld 重建器。
  - `vggt`: VGGT 重建器。

### 3. BEV 生成 (Bird's Eye View)
基于几何反投影与重力对齐算法，将重建场景转换为标准的正交俯视图。
- **核心功能**:
  - 自动地平面检测与重力对齐。
  - 基于 DSM (Digital Surface Model) 的遮挡处理与纹理映射。
  - 生成 `scene_metadata.npz` 用于后续任务。

### 4. 视图合成 (Crafter / ViewCrafter)
基于 ViewCrafter 的视频生成与新视角合成管线，支持从稀疏视角生成连贯的漫游视频。
- **特点**:
  - **直接支持 COLMAP**: 无需中间转换，直接读取 `sparse/` 和 `depths/` 文件夹。
  - **显存优化**: 针对 H100 等硬件进行了大分辨率渲染优化。
  - **去 Dust3r 依赖**: 纯净的推理管线，减少环境依赖。

### 5. 3D Gaussian Splatting
集成 3DGS 训练与渲染管线，用于高质量的实进渲染。

---

## 🛠️ 环境安装

本项目推荐使用 `micromamba` 或 `conda` 进行环境管理。

```bash
# 创建并激活环境 (以 micromamba 为例)
micromamba create -n wd python=3.10
micromamba activate wd

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
pip install gradio==6.2.0
pip install huggingface-hub[torch]==0.23.5
```

## 📂 目录结构说明

```text
wd_projects/
├── bev/                # BEV 生成核心代码
├── checkpoints/        # 模型权重存放路径
├── crafter/            # ViewCrafter 视图合成模块 (Refactored)
├── data/               # 数据存放目录
├── gaussians/          # 3D Gaussian Splatting 模块
├── reconstruction/     # 多种重建算法的统一接口
├── undistortion/       # 图像去畸变模块
├── scripts/            # 实用脚本工具
└── requirements.txt    # 项目依赖
```
