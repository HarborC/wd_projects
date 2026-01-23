# BEV Reconstruction 工具文档

本文档详细介绍了 `bev_reconstruction.py` 工具的原理、使用方法以及输入输出数据格式。该工具旨在通过多视角几何重建数据（COLMAP）和深度图，生成高质量的场景俯视图（BEV, Bird's Eye View），并提供对齐后的场景元数据供后续算法使用。

## 1. 算法原理

本工具生成 BEV 的核心流程是 **"基于几何的反向投影与色彩融合"**。主要步骤如下：

### 1.1 3D 点云提升 (Lifting)
利用 COLMAP 重建得到的相机位姿（Extrinsics）和内参（Intrinsics），结合单目深度估计得到的深度图（Depth Maps），将每一张 2D 图像的像素反向投影到 3D 空间，构建出场景的 **稠密点云 (Dense Point Cloud)**。

### 1.2 重力对齐 (Gravity Alignment)
COLMAP 重建的世界坐标系通常是任意的。为了生成标准的俯视图，必须找到“地面”并将坐标系摆正。
*   **地面检测**：使用 SegFormer 语义分割模型识别图像中的“地面”像素，提取对应的 3D 点。
*   **平面对齐**：使用 RANSAC 算法拟合地面平面方程，计算地面法向量。
*   **旋转校正**：计算旋转矩阵 $R_{align}$，将场景旋转，使得 Y 轴垂直于地面（本工具中设定 Y 轴负方向为“上”，即天空方向；Y 轴正方向为“下”，即地面方向）。

### 1.3 数字表面模型 (DSM) 生成
为了解决点云稀疏和遮挡问题，算法在对齐后的 X-Z 平面（俯视平面）上划分网格。
*   对于每个网格单元（即 BEV 的一个像素），寻找落入其中的所有点云中 **高度最高（Y 值最小）** 的点。
*   这形成了一个 **数字表面模型 (DSM)**，实际上是一个高度图，表示了从上往下看时场景表面的几何形状。
*   **空洞填充**：利用形态学闭运算填充室内封闭区域内的几何空洞。

### 1.4 反向投影与纹理映射 (Inverse Mapping & Texturing)
为了获得高质量的纹理，算法不直接拼接点云颜色，而是采用反向投影：
1.  遍历 BEV 图像的每个像素 $(u, v)$。
2.  根据 DSM 获取该位置的高度 $y$，重构出 3D 点 $P_{aligned} = (x, y, z)$。
3.  将 $P_{aligned}$ 逆变换回原始 COLMAP 坐标系。
4.  将该点投影到所有可见相机视图中。
5.  **遮挡剔除 (Occlusion Check)**：比对投影深度与深度图记录的深度，判断该点是否被遮挡。
6.  **色彩融合**：对所有通过遮挡测试的相机的颜色进行加权平均，得到该 BEV 像素的颜色。

---

## 2. 使用指南

### 运行命令
```bash
python -m bev.bev_reconstruction \
  --reconstruction_dir /path/to/reconstruction_output \
  --output_dir /path/to/output_folder \
  --target_size 2000 \
  --resolution 0.01
```

### 参数说明
*   `--reconstruction_dir`: **(必选)** 重建数据根目录，需包含 `sparse/`, `images/`, `depths/`。
*   `--output_dir`: **(必选)** 结果输出目录。
*   `--target_size`: **(可选, 默认 1000)** BEV 图像长边的目标像素大小。脚本会自动计算分辨率以适配此大小。设置为 0 则使用固定分辨率。
*   `--resolution`: **(可选, 默认 0.01)** 当 `target_size` 为 0 时使用的固定分辨率（单位：世界坐标系单位，通常不对应物理米，除非 COLMAP 已做尺度对齐）。

---

## 3. 输入数据说明

`reconstruction_dir` 必须遵循以下文件结构：

```text
reconstruction_dir/
├── sparse/
│   └── 0/              # COLMAP 稀疏重建模型 (cameras.bin, images.bin, points3D.bin)
├── images/             # 原始 RGB 图像
│   ├── frame_0000.jpg
│   └── ...
└── depths/             # 对应的深度图 (.npy 格式)
    ├── frame_0000.npy
    └── ...
```

---

## 4. 输出结果说明

输出目录 `output_dir` 结构如下：

```text
output_dir/
├── bev_images/
│   ├── merged_bev_world_color_corrected.png  # 【最终结果】生成的 RGB BEV 图像
│   └── bev_diagnostic_mask.png               # 诊断掩码（绿色=有效，红色=推测填充，灰色=无数据）
├── depths_original_res/                      # 备份的原始分辨率深度图 (.npy)
├── scene_metadata.npz                        # 【核心】场景元数据及坐标转换参数
├── reconstructed_pointcloud_world_raw.ply     # 原始点云（未对齐）
└── reconstructed_pointcloud_world_aligned.ply # 对齐后的点云（Y轴垂直地面）
```

### 4.1 核心元数据：`scene_metadata.npz`

这是一个 NumPy 压缩文件，包含了将其他数据映射到 BEV 坐标系所需的所有信息。

**包含的字段 (Keys)：**

| 字段名 | 形状 (Shape) | 说明 |
| :--- | :--- | :--- |
| `intrinsics` | $(N, 3, 3)$ | 相机内参矩阵 $K$ |
| `extrinsics` | $(N, 4, 4)$ | **已对齐的**世界坐标系到相机的变换矩阵 $T_{world\_aligned \to cam}$ |
| `bev_params` | $(6,)$ | BEV 投影参数 `[x_min, z_max, res, ground_level, bev_w, bev_h]` |
| `depth_paths` | $(N,)$ | 深度文件路径列表 |
| `img_paths` | $(N,)$ | 图片文件路径列表 |

### 4.2 坐标转换指南

如何使用 `bev_params` 在 **已对齐的世界坐标** 与 **BEV 像素坐标** 之间转换：

假设 `bev_params = [x_min, z_max, res, ground, w, h]`

**1. 世界坐标 $(x, z) \to$ BEV 像素 $(u, v)$**
```python
u = (x - x_min) / res
v = (z_max - z) / res  # 注意：图像Y轴向下，世界Z轴在上
```

**2. BEV 像素 $(u, v) \to$ 世界坐标 $(x, z)$**
```python
x = x_min + (u + 0.5) * res
z = z_max - (v + 0.5) * res
```

**注意：**
`scene_metadata.npz` 中的 `extrinsics` 是基于 **已对齐坐标系** 的。这意味着，如果您有一个在此坐标系下的 3D 点 $P$，您可以直接使用 $P_{cam} = \text{extrinsics}[i] \times P$ 将其投影回相机图像，或使用上述公式将其投影到 BEV 图像，两者在几何上是完全一致的。
