# Pipeline 重构设计文档

**日期**: 2025-01-18
**作者**: Claude + User
**状态**: 已批准

## 1. 概述

### 1.1 目标

重构现有的 `pipeline.py`，将其从简单的函数式结构改造为可扩展的类架构，支持：

- 多种去畸变算法（GeoCalib, AnyCalib）
- 多种重建算法（DA3, MASt3R, HunyuanWorld, VGGT）
- 便于未来添加新步骤（渲染、评估等）
- 配置文件驱动

### 1.2 设计原则

1. **Stage 模式**：每个处理步骤都是一个独立的 Stage 类
2. **配置驱动**：通过字典或配置文件驱动 Pipeline 行为
3. **状态隔离**：每个 Stage 的输出和中间结果独立管理
4. **可扩展性**：新 Stage 可轻松添加而不影响现有代码

---

## 2. 架构设计

### 2.1 类结构

```
Pipeline (主流程编排器)
├── UndistortStage (去畸变阶段)
│   ├── GeoCalibUndistorter
│   └── AnyCalibUndistorter
├── ReconstructionStage (重建阶段)
│   ├── DA3Reconstructor
│   ├── MASt3RReconstructor
│   ├── HunyuanWorldReconstructor
│   └── VGGTReconstructor
└── [未来扩展] RenderStage, EvaluateStage...
```

### 2.2 目录结构

```
wd_projects/
├── pipeline.py                 # 主入口（CLI）
├── pipeline/
│   ├── __init__.py
│   ├── base.py                 # BaseStage, StageOutput
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── undistort_stage.py  # UndistortStage
│   │   └── reconstruction_stage.py
│   └── pipeline.py             # Pipeline 主类
├── undistortion/               # 现有去畸变模块
├── reconstruction/             # 现有重建模块
└── configs/                    # 配置文件目录
    ├── default.json
    ├── hunyuanworld.json
    └── vggtx.json
```

---

## 3. 核心接口设计

### 3.1 BaseStage

```python
class BaseStage(ABC):
    """所有处理步骤的基类"""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self, input_dir: str, output_dir: str) -> StageOutput:
        """执行该步骤，返回输出信息"""
        pass

    def validate_input(self, input_dir: str) -> bool:
        """验证输入是否有效"""
        return Path(input_dir).exists()

    def prepare_output(self, output_dir: str):
        """准备输出目录"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
```

### 3.2 StageOutput

```python
class StageOutput:
    """Stage 输出数据容器"""
    def __init__(self, output_dir: str, metadata: dict = None):
        self.output_dir = output_dir
        self.metadata = metadata or {}
```

### 3.3 UndistortStage

```python
class UndistortStage(BaseStage):
    def __init__(self, config: dict):
        super().__init__(config)
        method = config.get("method", "geocalib")

        if method == "geocalib":
            self.undistorter = GeoCalibUndistorter()
        elif method == "anycalib":
            self.undistorter = AnyCalibUndistorter()
        else:
            raise ValueError(f"Unknown undistort method: {method}")

    def run(self, input_dir: str, output_dir: str) -> StageOutput:
        self.prepare_output(output_dir)
        intrinsics_path = self.config.get("intrinsics_path")

        self.undistorter.process_directory(input_dir, output_dir, intrinsics_path)

        return StageOutput(
            output_dir=output_dir,
            metadata={"intrinsics": intrinsics_path}
        )
```

### 3.4 ReconstructionStage

```python
class ReconstructionStage(BaseStage):
    def __init__(self, config: dict):
        super().__init__(config)
        self.reconstructor = ReconstructionFactory.create(
            backend=config.get("method", "da3"),
            device=config.get("device")
        )

    def run(self, input_dir: str, output_dir: str) -> StageOutput:
        self.prepare_output(output_dir)
        self.reconstructor.process_directory(input_dir, output_dir)
        return StageOutput(output_dir=output_dir)
```

---

## 4. Pipeline 类

```python
class Pipeline:
    """
    多视角重建处理流程的主编排器。
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.stages = {}
        self.stage_outputs = {}

        self._build_stages()

    def _build_stages(self):
        """根据配置构建各个 Stage"""
        if "undistort" in self.config:
            self.stages["undistort"] = UndistortStage(self.config["undistort"])

        if "reconstruction" in self.config:
            self.stages["reconstruction"] = ReconstructionStage(
                self.config["reconstruction"]
            )

    def run(self, input_dir: str, output_base_dir: str) -> dict:
        """执行完整流程"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Pipeline")
        self.logger.info("=" * 60)

        current_input = input_dir

        stage_order = ["undistort", "reconstruction"]
        for stage_name in stage_order:
            if stage_name not in self.stages:
                continue

            stage = self.stages[stage_name]
            output_dir = str(Path(output_base_dir) / stage.config["output_dir"])

            self.logger.info(f"\n>>> Running stage: {stage_name}")
            try:
                output = stage.run(current_input, output_dir)
                self.stage_outputs[stage_name] = output
                current_input = output_dir
                self.logger.info(f"<<< Stage {stage_name} completed")
            except Exception as e:
                self.logger.error(f"Stage {stage_name} failed: {e}")
                raise

        self.logger.info("Pipeline completed successfully")
        return self.stage_outputs

    @classmethod
    def from_config_file(cls, config_path: str) -> "Pipeline":
        """从配置文件创建 Pipeline"""
        import json
        with open(config_path) as f:
            config = json.load(f)
        return cls(config)
```

---

## 5. 命令行接口

### 5.1 参数设计

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input-dir` | str | 必需 | 输入图像目录 |
| `--output-dir` | str | 必需 | 输出根目录 |
| `--undistort-method` | str | geocalib | 去畸变方法 |
| `--skip-undistort` | flag | False | 跳过去畸变 |
| `--reconstruct-method` | str | da3 | 重建方法 |
| `--device` | str | auto | 计算设备 |
| `--config` | str | None | 配置文件路径 |

### 5.2 使用示例

```bash
# 命令行方式
python pipeline.py \
    --input-dir ./data/images \
    --output-dir ./output \
    --undistort-method anycalib \
    --reconstruct-method hunyuanworld

# 配置文件方式
python pipeline.py \
    --input-dir ./data/images \
    --output-dir ./output \
    --config configs/hunyuanworld.json

# 跳过去畸变
python pipeline.py \
    --input-dir ./data/undistorted \
    --output-dir ./output \
    --skip-undistort \
    --reconstruct-method vggtx
```

---

## 6. 配置文件格式

### 6.1 示例配置

```json
{
    "undistort": {
        "method": "geocalib",
        "output_dir": "undistorted",
        "intrinsics_path": "undistorted/intrinsics.json"
    },
    "reconstruction": {
        "method": "hunyuanworld",
        "device": "cuda",
        "output_dir": "reconstruction"
    }
}
```

---

## 7. 实现计划

### 7.1 需要创建的文件

1. `pipeline/__init__.py` - 包初始化
2. `pipeline/base.py` - 基础类定义
3. `pipeline/stages/__init__.py` - stages 包
4. `pipeline/stages/undistort_stage.py` - 去畸变 stage
5. `pipeline/stages/reconstruction_stage.py` - 重建 stage
6. `pipeline/pipeline.py` - Pipeline 主类
7. 更新 `pipeline.py` - 新的 CLI 入口
8. `configs/` - 配置文件目录

### 7.2 实现步骤

1. 创建目录结构
2. 实现 `base.py`
3. 实现 Stage 类
4. 实现 Pipeline 类
5. 更新 CLI 入口
6. 创建配置文件
7. 测试

---

## 8. 支持的算法

### 8.1 去畸变方法

| 方法 | 类名 | 状态 |
|------|------|------|
| geocalib | GeoCalibUndistorter | 现有 |
| anycalib | AnyCalibUndistorter | 现有 |

### 8.2 重建方法

| 方法 | 类名 | 状态 |
|------|------|------|
| da3 | DA3Reconstructor | 现有 |
| mast3r | MASt3RReconstructor | 现有 |
| hunyuanworld | HunyuanWorldReconstructor | 现有 |
| vggtx | VGGTReconstructor | 现有 |

---

## 9. 未来扩展

### 9.1 可能的 Stage

- `RenderStage` - Gaussian Splatting 渲染
- `EvaluateStage` - 质量评估
- `ExportStage` - 导出不同格式

### 9.2 扩展方式

```python
# 注册新 Stage
class CustomStage(BaseStage):
    def run(self, input_dir: str, output_dir: str) -> StageOutput:
        # 实现逻辑
        pass

# 在 Pipeline 中使用
pipeline.stages["custom"] = CustomStage(config)
```

---

## 10. 变更历史

| 日期 | 变更 |
|------|------|
| 2025-01-18 | 初始设计文档 |
