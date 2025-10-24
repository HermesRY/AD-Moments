# AD-Moments

本仓库包含 AD-Moments 项目源码与实验数据，核心实现为 CTMS（Circadian-Task-Movement-Social）时序编码器及融合与检测流水线。

快速说明

- 主要目录
  - `models/`：CTMS 模型实现（`ctms_model.py`）、GPU 加速版本（`ctms_model_gpu.py`）、训练/搜索/实验 notebooks 与导出结果
  - `figures/`：用于论文的图像资源
  - `data/` 或 `models/abnormalTS/`：部分示例数据与异常时间戳（若有）
  - `main.tex`：论文稿（已根据实现同步修改部分方法描述）

- 依赖（示例）
  - Python 3.10+
  - PyTorch 2.x
  - 必要包请见 `models/requirements.txt` 或仓库根目录中的 `requirements.txt`（若存在）

快速开始（在有 GPU 的工作站上）

1. 创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r models/requirements.txt
```

2. 小规模试跑模型（示例）

```bash
# 在 models 目录下运行测试脚本或 notebook
cd Moments/models
python usage_example.py   # 若存在
```

3. 运行 notebook

请使用 Jupyter Lab / Notebook 打开 `models/CTMS_train_with_optimal_config.ipynb` 或 `models/optimized_CTMS_search_gpu.ipynb` 进行实验复现。

关于复现与注意事项

- 论文中方法已根据代码实现进行了小幅调整（例如 Task encoder 使用序列模板与滑动子序列的 DTW 对齐，GPU 运行时可选 pooled-cosine 近似以加速）。详见 `models/main.tex` 中的实现说明段落。
- 仓库中包含部分训练好的模型文件（.pt）与大型 CSV/PNG 文件。如果你不希望这些大文件出现在历史中，建议使用 Git LFS 或把模型文件作为 release 附件。

联系方式

如需进一步帮助或希望我为你准备可运行的最小示例（含小数据并能快速复现），请在 issue 中说明或直接联系维护者。
