# AD-Moments

本仓库包含 AD-Moments 项目源码与实验数据，核心实现为 CTMS（Circadian-Task-Movement-Social）时序编码器及融合与检测流水线。

快速说明

- 主要目录
  - `models/`：CTMS 模型实现（`ctms_model.py`）、GPU 加速版本（`ctms_model_gpu.py`）、训练/搜索/实验 notebooks 与导出结果
  - `figures/`：用于论文的图像资源
  - `data/` 或 `models/abnormalTS/`：部分示例数据与异常时间戳（若有）
  - `main.tex`：论文稿（已根据实现同步修改部分方法描述）
