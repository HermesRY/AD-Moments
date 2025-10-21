# Exp2 异常片段时间戳检测功能

## 功能概述

在 `exp2_daily_patterns.py` 中添加了异常片段检测和时间戳记录功能，能够：
1. 检测异常评分超过阈值的时间窗口
2. 记录每个异常片段的精确时间戳（start_time, end_time）
3. 统计每个受试者的异常片段数量和比例
4. 输出详细的JSON文件供后续分析

## 实现说明

### 1. 修改了 `compute_subject_score` 函数

**新增参数:**
- `return_timestamps` (bool): 如果为True，返回 (scores, timestamps) 元组

**返回值:**
- 当 `return_timestamps=False`: 返回 np.array 的分数
- 当 `return_timestamps=True`: 返回 (scores, timestamps) 元组
  - `scores`: np.array of anomaly scores
  - `timestamps`: list of dicts，每个dict包含:
    - `window_idx`: 窗口索引
    - `start_time`: 窗口开始时间点
    - `end_time`: 窗口结束时间点

**时间计算方法:**
```python
window_idx = i + j  # 第几个窗口
start_time = window_idx * STRIDE  # 开始时间点
end_time = start_time + SEQ_LEN   # 结束时间点
```

### 2. 新增 `detect_abnormal_segments` 函数

**功能:**
- 检测并记录所有异常片段
- 使用百分位数(默认95th)作为异常阈值

**参数:**
- `subjects`: 受试者列表
- `model`: CTMS模型
- `baseline_stats`: 基线统计数据
- `threshold_percentile`: 异常阈值百分位数(默认95)

**返回值:**
```python
{
    'threshold': float,  # 异常阈值
    'threshold_percentile': int,  # 使用的百分位数
    'total_subjects_analyzed': int,  # 分析的总受试者数
    'subjects_with_abnormalities': int,  # 有异常的受试者数
    'abnormal_records': [  # 每个受试者的异常记录
        {
            'subject_id': int,
            'label': str,  # 'CN' or 'CI'
            'total_windows': int,  # 总窗口数
            'abnormal_windows': int,  # 异常窗口数
            'abnormal_percentage': float,  # 异常百分比
            'mean_score': float,  # 平均分数
            'max_score': float,  # 最大分数
            'abnormal_segments': [  # 所有异常片段
                {
                    'window_idx': int,
                    'start_time': int,
                    'end_time': int,
                    'score': float
                },
                ...
            ]
        },
        ...
    ]
}
```

### 3. 主函数更新

在主函数中添加了异常检测步骤:

```python
# 5. Detecting abnormal segments with timestamps...
print("   Analyzing CI subjects...")
ci_abnormal = detect_abnormal_segments(ci_subjects, model, baseline_stats, threshold_percentile=95)

print("   Analyzing CN test subjects...")
cn_abnormal = detect_abnormal_segments(cn_test, model, baseline_stats, threshold_percentile=95)
```

### 4. 输出文件

生成两个新的JSON文件:

**Without_Personalization/outputs/exp2_abnormal_segments_CI.json**
- CI组受试者的异常片段详细记录

**Without_Personalization/outputs/exp2_abnormal_segments_CN.json**
- CN测试组受试者的异常片段详细记录

## 输出示例

```json
{
  "threshold": 2.3456,
  "threshold_percentile": 95,
  "total_subjects_analyzed": 36,
  "subjects_with_abnormalities": 24,
  "abnormal_records": [
    {
      "subject_id": 1,
      "label": "CI",
      "total_windows": 100,
      "abnormal_windows": 8,
      "abnormal_percentage": 8.0,
      "mean_score": 1.8234,
      "max_score": 3.4567,
      "abnormal_segments": [
        {
          "window_idx": 15,
          "start_time": 150,
          "end_time": 180,
          "score": 2.8901
        },
        {
          "window_idx": 42,
          "start_time": 420,
          "end_time": 450,
          "score": 3.4567
        }
      ]
    }
  ]
}
```

## 使用说明

### 基本运行:
```bash
cd /path/to/Exp_Full/Without_Personalization
python3 exp2_daily_patterns.py
```

### 自定义阈值:
可在代码中修改 `threshold_percentile` 参数:
```python
ci_abnormal = detect_abnormal_segments(ci_subjects, model, baseline_stats, threshold_percentile=90)
```

### 读取结果:
```python
import json

# 读取CI组异常记录
with open('outputs/exp2_abnormal_segments_CI.json', 'r') as f:
    ci_abnormal = json.load(f)

# 查看有多少CI受试者有异常
print(f"CI subjects with abnormalities: {ci_abnormal['subjects_with_abnormalities']}/{ci_abnormal['total_subjects_analyzed']}")

# 查看特定受试者的异常片段
for record in ci_abnormal['abnormal_records']:
    if record['subject_id'] == 1:
        print(f"Subject 1 has {record['abnormal_windows']} abnormal windows")
        for seg in record['abnormal_segments']:
            print(f"  Window {seg['window_idx']}: time {seg['start_time']}-{seg['end_time']}, score={seg['score']:.3f}")
```

## 技术细节

### 时间窗口参数:
- `SEQ_LEN = 30`: 每个窗口长度为30个时间点
- `STRIDE = 10`: 窗口滑动步长为10个时间点
- 窗口有70%重叠 (overlap = 20/30 = 66.7%)

### 异常判定:
- 使用所有窗口分数的95th百分位数作为阈值
- 分数超过阈值的窗口被标记为异常

### 两次遍历策略:
1. **第一遍**: 收集所有分数，计算阈值
2. **第二遍**: 检测异常窗口，记录时间戳

## 应用场景

1. **临床分析**: 识别CI患者的异常行为时间段
2. **模式发现**: 分析异常片段是否集中在特定时间段
3. **个体化评估**: 为每个受试者生成异常报告
4. **时间序列分析**: 研究异常片段的时间分布规律

## 注意事项

⚠️ **当前状态**: 代码已完成并测试，但需要确保数据集包含以下字段:
- `circadian_rhythm`: 昼夜节律特征
- `task_completion`: 任务完成特征  
- `movement_patterns`: 运动模式特征
- `social_interactions`: 社交互动特征

如果数据集只有原始的 `sequence` 字段，需要先运行特征提取预处理步骤。

## 两个版本

✅ **Without_Personalization**: 使用80/20分割，seed=0
✅ **With_Personalization**: 使用70/30分割，seed=1

两个版本都已添加相同的时间戳检测功能。
