# Reconstruction → SEVA 转换器使用指南

## 功能说明

将你的 Reconstruction 模块输出转换为 Stable Virtual Camera (SEVA) 所需的格式。

## 输入/输出格式

### 输入（Reconstruction输出）
```
recon_output/
├── images/              # 输入图像
│   ├── img_000.jpg
│   ├── img_001.jpg
│   └── ...
└── sparse/0/            # COLMAP格式
    ├── cameras.bin      # 相机内参
    ├── images.bin       # 相机位姿
    └── points3D.ply     # 点云（可选）
```

### 输出（SEVA格式）
```
seva_output/
├── transforms.json              # 相机参数（所有图像共享）
├── train_test_split_1.json      # 单视图分割
├── train_test_split_3.json      # 3视图分割
├── train_test_split_6.json      # 6视图分割
├── train_test_split_9.json      # 9视图分割
└── images/                      # 图像（复制或符号链接）
    ├── img_000.jpg
    └── ...
```

## 安装依赖

```bash
pip install pycolmap numpy pillow
```

## 使用方法

### 基本用法

```bash
python recon_to_seva_converter.py \
    --recon-output /path/to/reconstruction/output \
    --seva-output /path/to/seva/output
```

### 完整参数示例

```bash
python recon_to_seva_converter.py \
    --recon-output ./test_reconstruction_output \
    --seva-output ./seva_input_data \
    --num-splits 1,3,6,9,16,32 \
    --split-strategy uniform \
    --copy-images
```

### 使用符号链接（节省磁盘空间）

```bash
python recon_to_seva_converter.py \
    --recon-output ./test_reconstruction_output \
    --seva-output ./seva_input_data \
    --symlink-images
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--recon-output` | Reconstruction输出目录路径 | 必需 |
| `--seva-output` | SEVA格式输出目录路径 | 必需 |
| `--num-splits` | 逗号分隔的训练视图数量 | `"1,3,6,9"` |
| `--split-strategy` | 训练索引选择策略 | `"uniform"` |
| `--copy-images` | 复制图像到输出目录 | False |
| `--symlink-images` | 创建符号链接（覆盖--copy-images） | False |

### split-strategy 选项

- **`uniform`**: 均匀分布选择训练图像（推荐）
- **`first`**: 选择前N张图像
- **`random`**: 随机选择（固定种子42，可复现）

## 完整工作流示例

### 步骤1: 运行Reconstruction

```bash
# 使用MASt3R
python -m reconstruction.factory \
    --input-dir ./my_images \
    --output-dir ./recon_output \
    --backend mast3r \
    --device cuda
```

### 步骤2: 转换为SEVA格式

```bash
python recon_to_seva_converter.py \
    --recon-output ./recon_output \
    --seva-output ./seva_input \
    --num-splits 1,3,6,9 \
    --split-strategy uniform \
    --symlink-images
```

### 步骤3: 运行SEVA生成新视角

```bash
cd test_code/stable-virtual-camera

# 单视图测试
python demo.py \
    --data_path ../../seva_input \
    --num_inputs 1 \
    --video_save_fps 10

# 3视图测试
python demo.py \
    --data_path ../../seva_input \
    --num_inputs 3 \
    --video_save_fps 10
```

## transforms.json 格式说明

```json
{
    "w": 960.0,                    // 图像宽度
    "h": 540.0,                    // 图像高度
    "fl_x": 434.95,                // 焦距X
    "fl_y": 433.87,                // 焦距Y
    "cx": 480.0,                   // 主点X
    "cy": 270.0,                   // 主点Y
    "k1": 0.03,                    // 畸变系数（可选）
    "k2": -0.02,                   // 畸变系数（可选）
    "camera_model": "PINHOLE",     // 相机模型
    "frames": [                    // 每个图像的位姿
        {
            "file_path": "images/img_000.jpg",
            "transform_matrix": [[4x4矩阵]],  // world-to-camera
            "colmap_im_id": 1
        },
        ...
    ]
}
```

## train_test_split_N.json 格式说明

```json
{
    "train_ids": [0, 50, 100],    // 训练图像索引
    "test_ids": [1,2,3,...]       // 测试图像索引
}
```

## 故障排除

### 错误: "Sparse directory not found"
确保Reconstruction已成功完成，并且 `sparse/0/` 目录存在。

### 错误: "pycolmap not available"
```bash
pip install pycolmap
```

### 错误: "No cameras found in reconstruction"
检查COLMAP输出是否完整，运行Reconstruction时是否成功。

## 注意事项

1. **相机模型转换**: COLMAP使用OpenGL约定（world-to-camera），SEVA也使用相同约定
2. **图像顺序**: 转换器会按colmap_im_id排序图像
3. **符号链接**: 在Windows上可能不可用，使用 `--copy-images` 代替
4. **内存**: 大数据集可能需要较多内存来读取COLMAP模型

## 输出文件验证

转换完成后，验证输出：

```bash
# 检查文件结构
ls -la seva_input/

# 检查transforms.json
cat seva_input/transforms.json | head -50

# 检查分割文件
cat seva_input/train_test_split_3.json
```
